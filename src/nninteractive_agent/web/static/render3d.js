// The 3D view: volume rendering and object surfaces in one WebGL2 canvas, composited by depth.
//
// Each frame: (1) opaque surfaces are drawn with 4x MSAA into an offscreen colour + depth target;
// (2) semi-transparent surfaces (the object being edited, or all with the opacity slider below 1)
// are blended into a second target, depth-tested against the opaque ones, keeping the depth of the
// nearest; (3) a full-screen ray caster marches through the volume from the eye up to the nearest
// opaque surface and composites the volume over the surfaces, the semi-transparent layer where the
// ray reaches it (so bone in front of an organ hides it, and the organ hides what lies behind it),
// and outlines the surfaces.
//
// World space is NIfTI RAS mm, the space of the surface vertices. The camera orbits a centre:
// azimuth 180 / elevation 0 looks at the patient from the front (anterior), with the patient's
// right on the left of the screen, as in the slice views.

const DEG = Math.PI / 180
const FOV = 30 * DEG
// Lights in camera space (x right, y up, z towards the viewer): key from the upper left, fill from
// the right. Surfaces and volume use the same lights.
const KEY = norm([-0.45, 0.6, 0.65])
const FILL = norm([0.65, -0.1, 0.75])
const BG_TOP = [0.105, 0.122, 0.153] // sRGB
const BG_BOTTOM = [0.024, 0.028, 0.035]
const MAX_STEPS = 4096
const MAX_PIXELS = 1.6e6

function norm(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l) }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
function axpy(y, a, x) { return [y[0] + a * x[0], y[1] + a * x[1], y[2] + a * x[2]] }
const srgbToLinear = (c) => c.map((x) => Math.pow(x, 2.2))

// 4x4 matrices, column-major (as WebGL wants them).
function mul(a, b) {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
      o[c * 4 + r] = s
    }
  }
  return o
}

function invert(m) {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  const d = det ? 1 / det : 0
  return new Float32Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * d, (a02 * b10 - a01 * b11 - a03 * b09) * d,
    (a31 * b05 - a32 * b04 + a33 * b03) * d, (a22 * b04 - a21 * b05 - a23 * b03) * d,
    (a12 * b08 - a10 * b11 - a13 * b07) * d, (a00 * b11 - a02 * b08 + a03 * b07) * d,
    (a32 * b02 - a30 * b05 - a33 * b01) * d, (a20 * b05 - a22 * b02 + a23 * b01) * d,
    (a10 * b10 - a11 * b08 + a13 * b06) * d, (a01 * b08 - a00 * b10 - a03 * b06) * d,
    (a30 * b04 - a31 * b02 + a33 * b00) * d, (a21 * b02 - a20 * b04 - a23 * b00) * d,
    (a11 * b07 - a10 * b09 - a12 * b06) * d, (a00 * b09 - a01 * b07 + a02 * b06) * d,
    (a31 * b01 - a30 * b03 - a32 * b00) * d, (a20 * b03 - a21 * b01 + a22 * b00) * d,
  ])
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far)
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0])
}

/** In-place separable [1 4 6 4 1] / 16 blur (sigma about 1 voxel) of an x-fastest volume. */
function blur3d(f, [nx, ny, nz]) {
  const tmp = new Float32Array(Math.max(nx, ny, nz))
  const strides = [1, nx, nx * ny], sizes = [nx, ny, nz]
  for (let axis = 0; axis < 3; axis++) {
    const n = sizes[axis], st = strides[axis]
    if (n < 3) continue
    const others = [0, 1, 2].filter((a) => a !== axis)
    const [na, nb] = others.map((a) => sizes[a]), [sa, sb] = others.map((a) => strides[a])
    for (let b = 0; b < nb; b++) {
      for (let a = 0; a < na; a++) {
        const base = a * sa + b * sb
        for (let i = 0; i < n; i++) {
          const at = (k) => f[base + Math.min(n - 1, Math.max(0, k)) * st]
          tmp[i] = (at(i - 2) + 4 * at(i - 1) + 6 * at(i) + 4 * at(i + 1) + at(i + 2)) / 16
        }
        for (let i = 0; i < n; i++) f[base + i * st] = tmp[i]
      }
    }
  }
  return f
}

/** Area-weighted vertex normals of a triangle mesh (triangles wound counter-clockwise from outside). */
export function vertexNormals(pos, tris) {
  const n = new Float32Array(pos.length)
  for (let f = 0; f < tris.length; f += 3) {
    const a = 3 * tris[f], b = 3 * tris[f + 1], c = 3 * tris[f + 2]
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2]
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2]
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    for (const i of [a, b, c]) { n[i] += nx; n[i + 1] += ny; n[i + 2] += nz }
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1
    n[i] /= l; n[i + 1] /= l; n[i + 2] /= l
  }
  return n
}

// ------------------------------------------------------------------------------------ shaders
const MESH_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uView, uProj;
out vec3 vN;
out vec3 vP;
void main() {
  vec4 p = uView * vec4(aPos, 1.0);
  vP = p.xyz;
  vN = mat3(uView) * aNormal;
  gl_Position = uProj * p;
}`

const MESH_FS = `#version 300 es
precision highp float;
in vec3 vN;
in vec3 vP;
uniform vec3 uColor; // linear
uniform float uOpacity;
uniform vec3 uKey, uFill; // camera space
out vec4 outColor;
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(-vP);
  // (linear light) hemisphere ambient, a wrapped key light, a fill light, specular and a soft rim
  vec3 amb = mix(vec3(0.035, 0.033, 0.04), vec3(0.11, 0.12, 0.14), 0.5 + 0.5 * n.y);
  float key = max((dot(n, uKey) + 0.25) / 1.25, 0.0);
  float fill = max(dot(n, uFill), 0.0);
  float spec = pow(max(dot(n, normalize(uKey + v)), 0.0), 60.0);
  float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  vec3 base = uColor * 0.78; // headroom, so lit sides of bright colours do not all clip to one flat tone
  vec3 c = base * (amb + 1.0 * key + 0.2 * fill) + 0.25 * spec + 0.06 * rim * vec3(0.75, 0.85, 1.0);
  outColor = vec4(pow(max(c, 0.0), vec3(1.0 / 2.2)), uOpacity);
}`

const QUAD_VS = `#version 300 es
out vec2 vNdc;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  vNdc = p;
  gl_Position = vec4(p, 0.0, 1.0);
}`

const COMPOSITE_FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
uniform mat4 uInvViewProj;
uniform vec3 uEye;
uniform mat4 uWorld2Tex;
uniform mat3 uGradToWorld;
uniform vec3 uTexel;       // 1 / volume size
uniform float uStep;       // ray step in voxels
uniform float uRefLen;     // the transfer function's opacity is per this length (mm)
uniform int uMode;         // 0 no volume, 1 shaded, 2 maximum intensity projection
uniform sampler3D uVol;
uniform sampler3D uSmooth; // the volume blurred (sigma 1.4 voxels): normals without the voxel grid
uniform sampler2D uTF;     // rows: colour (sRGB) + opacity, x: how much the colour is shaded
uniform highp sampler2D uPre; // running integrals of the transfer function over intensity (rgb colour
                              // x extinction, a extinction)
uniform float uPreN;
uniform sampler2D uMeshColor, uMeshDepth;
uniform sampler2D uLayerColor, uLayerDepth; // semi-transparent surfaces (premultiplied), their nearest depth
uniform int uLayer;                         // 1: there are semi-transparent surfaces
uniform vec2 uViewport;
uniform vec3 uBgTop, uBgBottom;         // linear
uniform vec3 uKeyW, uFillW;             // world space
uniform vec4 uLight;                    // ambient, diffuse, specular, shininess
uniform vec2 uGrad;                     // gradient magnitudes (per mm) from no to full shading
uniform float uXray;                    // how much surfaces behind the volume show through
uniform vec4 uGlass;                    // glass skin colour (linear) and opacity at its edges; a = 0: none
uniform vec2 uGlassIso;                 // x: the skin's intensity, y: rays must start below (in air)
uniform float uOutline;
uniform vec2 uNearFar;
out vec4 outColor;

vec3 toLin(vec3 c) { return pow(c, vec3(2.2)); }

float hash(vec2 p) { // Dave Hoskins' hash12
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

const float ROW_TF = 0.25, ROW_MAT = 0.75;

vec4 prefixAt(float s) {
  float x = clamp(s, 0.0, 1.0) * (uPreN - 1.0);
  float i = floor(x);
  vec4 a = texelFetch(uPre, ivec2(int(i), 0), 0);
  vec4 b = texelFetch(uPre, ivec2(int(min(i + 1.0, uPreN - 1.0)), 0), 0);
  return mix(a, b, x - i);
}

vec3 smoothNormal(vec3 p, vec3 dir, out float gm) { // facing the viewer; gm: gradient magnitude per mm
  vec3 g = vec3(
    texture(uSmooth, p + vec3(uTexel.x, 0.0, 0.0)).r - texture(uSmooth, p - vec3(uTexel.x, 0.0, 0.0)).r,
    texture(uSmooth, p + vec3(0.0, uTexel.y, 0.0)).r - texture(uSmooth, p - vec3(0.0, uTexel.y, 0.0)).r,
    texture(uSmooth, p + vec3(0.0, 0.0, uTexel.z)).r - texture(uSmooth, p - vec3(0.0, 0.0, uTexel.z)).r);
  g = uGradToWorld * (g / (2.0 * uTexel));
  gm = length(g);
  vec3 n = gm > 1e-8 ? -g / gm : -dir;
  return dot(n, dir) > 0.0 ? -n : n;
}

// The transfer function averaged over intensities sf..sb: rgb mean colour (linear), a mean
// extinction per uRefLen.
vec4 segment(float sf, float sb) {
  float m = 0.5 * (sf + sb);
  float h = max(0.5 * abs(sb - sf), 0.5 / uPreN);
  vec4 d = prefixAt(m + h) - prefixAt(m - h);
  return vec4(d.a > 1e-9 ? d.rgb / d.a : vec3(0.0), d.a / (2.0 * h));
}

float eyeDist(float d) { // depth buffer value -> distance along the view axis
  float z = d * 2.0 - 1.0;
  return 2.0 * uNearFar.x * uNearFar.y / (uNearFar.y + uNearFar.x - z * (uNearFar.y - uNearFar.x));
}

void main() {
  vec2 uv = gl_FragCoord.xy / uViewport;
  vec4 a = uInvViewProj * vec4(vNdc, -1.0, 1.0);
  vec4 b = uInvViewProj * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
  float vig = 1.0 - 0.25 * dot(vNdc * 0.5, vNdc * 0.5);
  vec3 bg = mix(uBgBottom, uBgTop, smoothstep(0.0, 1.0, uv.y)) * vig;

  // the opaque surfaces behind (premultiplied by their pixel coverage), outlined
  vec4 mc = texture(uMeshColor, uv);
  float md = texture(uMeshDepth, uv).r;
  float tMesh = 1e9;
  vec3 behind = bg;
  if (md < 1.0) {
    vec4 w = uInvViewProj * vec4(vNdc, md * 2.0 - 1.0, 1.0);
    tMesh = dot(w.xyz / w.w - uEye, dir);
    vec3 surf = toLin(mc.rgb / max(mc.a, 1e-4));
    if (uOutline > 0.0) {
      // darken the near side of depth discontinuities (silhouettes, one object in front of another)
      float z = eyeDist(md), edge = 0.0;
      vec2 px = 1.0 / uViewport;
      for (int k = 0; k < 4; k++) {
        vec2 o = vec2(k == 0 ? 1.0 : k == 1 ? -1.0 : 0.0, k == 2 ? 1.0 : k == 3 ? -1.0 : 0.0) * px * 1.5;
        float zn = eyeDist(texture(uMeshDepth, uv + o).r);
        edge = max(edge, smoothstep(0.012, 0.04, (zn - z) / z));
      }
      surf *= 1.0 - uOutline * edge;
    }
    behind = surf * mc.a + (1.0 - mc.a) * bg;
  }

  // the semi-transparent surfaces in front of them: one layer, composited where the ray reaches
  // the nearest of them, so the volume in front hides them as much as it hides opaque surfaces
  vec4 lay = vec4(0.0);
  float tLay = 1e9;
  if (uLayer > 0) {
    vec4 lc = texture(uLayerColor, uv);
    float ld = texture(uLayerDepth, uv).r;
    if (lc.a > 1e-3 && ld < 1.0) {
      vec4 w = uInvViewProj * vec4(vNdc, ld * 2.0 - 1.0, 1.0);
      tLay = dot(w.xyz / w.w - uEye, dir);
      lay = vec4(toLin(lc.rgb / lc.a) * lc.a, lc.a);
    }
  }
  bool layPending = lay.a > 0.0;

  vec4 acc = vec4(0.0);
  if (uMode > 0) {
    vec3 o = (uWorld2Tex * vec4(uEye, 1.0)).xyz;
    vec3 d = mat3(uWorld2Tex) * dir;
    vec3 dd = vec3(abs(d.x) < 1e-9 ? 1e-9 : d.x, abs(d.y) < 1e-9 ? 1e-9 : d.y, abs(d.z) < 1e-9 ? 1e-9 : d.z);
    vec3 t0s = -o / dd, t1s = (1.0 - o) / dd;
    vec3 tmin = min(t0s, t1s), tmax = max(t0s, t1s);
    float t0 = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
    float t1 = min(min(min(tmax.x, tmax.y), tmax.z), tMesh);
    if (layPending && tLay <= t0) { acc = lay; layPending = false; } // in front of the volume
    if (t1 > t0) {
      float stepMM = uStep / length(d / uTexel);
      float t = t0 + hash(gl_FragCoord.xy) * stepMM;
      if (uMode == 2) {
        float m = 0.0;
        for (int i = 0; i < ${MAX_STEPS}; i++) {
          if (t > t1) break;
          m = max(m, texture(uVol, o + d * t).r);
          t += stepMM;
        }
        vec4 tf = texture(uTF, vec2(m, ROW_TF));
        acc = vec4(toLin(tf.rgb) * tf.a, tf.a);
      } else {
        // Pre-integrated: each segment between two samples gets the transfer function integrated
        // over the values it passes (linear in between), so thin layers neither vanish between
        // samples nor show the step pattern as bands. Normals come from the segment's end.
        float tPrev = t0;
        float vPrev = texture(uVol, o + d * t0).r;
        // Glass skin: one surface layer where the ray first rises through the skin's intensity
        // (a layer of skin-range intensities instead would vary in thickness with the voxel grid
        // and show as bands). Only for rays that start outside the body.
        bool glass = uGlass.a > 0.0 && vPrev < uGlassIso.y;
        int empty = 0;
        bool last = false;
        float tEnd = layPending ? min(tLay, t1) : t1; // first up to the layer, then on to t1
        for (int i = 0; i < ${MAX_STEPS}; i++) {
          if (t >= tEnd) { t = tEnd; last = true; }
          vec3 p = o + d * t;
          float v = texture(uVol, p).r;
          float L = t - tPrev;
          if (glass && v >= uGlassIso.x) {
            if (L > 1.5 * stepMM) { empty = 0; last = false; t = tPrev + stepMM; continue; } // locate it finely
            glass = false;
            float tg = mix(tPrev, t, (uGlassIso.x - vPrev) / max(v - vPrev, 1e-6));
            float gmg;
            vec3 ng = smoothNormal(o + d * tg, dir, gmg);
            float facing = abs(dot(ng, dir));
            float a = uGlass.a * min(1.0, 0.08 + 1.6 * pow(1.0 - facing, 2.0)); // edges opaque, faces clear
            float diff = max(dot(ng, uKeyW), 0.0) + 0.3 * max(dot(ng, uFillW), 0.0);
            vec3 c = uGlass.rgb * (0.25 + 0.75 * diff) + 0.35 * pow(max(dot(ng, normalize(uKeyW - dir)), 0.0), 60.0);
            acc.rgb += (1.0 - acc.a) * a * c;
            acc.a += (1.0 - acc.a) * a;
          }
          vec4 seg = segment(vPrev, v);
          if (seg.a * L < 1e-5 * uRefLen) { // empty: after a few empty segments take longer ones
            vPrev = v;
            tPrev = t;
            if (last) {
              if (!layPending || tEnd >= t1) break;
              acc.rgb += (1.0 - acc.a) * lay.rgb;
              acc.a += (1.0 - acc.a) * lay.a;
              layPending = false;
              tEnd = t1;
              last = false;
            }
            empty++;
            t += (empty > 4 ? 3.0 : 1.0) * stepMM;
            continue;
          }
          if (L > 1.5 * stepMM) { empty = 0; last = false; t = tPrev + stepMM; continue; } // redo finely
          empty = 0;
          vec4 mat = texture(uTF, vec2(v, ROW_MAT));
          float gm;
          vec3 n = smoothNormal(p, dir, gm);
          float shade = mat.x * smoothstep(uGrad.x, uGrad.y, gm);
          vec3 base = seg.rgb;
          float diff = max(dot(n, uKeyW), 0.0) + 0.3 * max(dot(n, uFillW), 0.0);
          float spec = pow(max(dot(n, normalize(uKeyW - dir)), 0.0), uLight.w);
          vec3 lit = base * (uLight.x + uLight.y * diff) + uLight.z * spec;
          vec3 col = mix(base * (uLight.x + 0.55 * uLight.y), lit, shade);
          float alpha = 1.0 - exp(-seg.a * L / uRefLen);
          acc.rgb += (1.0 - acc.a) * alpha * col;
          acc.a += (1.0 - acc.a) * alpha;
          vPrev = v;
          tPrev = t;
          if (acc.a > 0.985) break;
          if (last) {
            if (!layPending || tEnd >= t1) break;
            acc.rgb += (1.0 - acc.a) * lay.rgb;
            acc.a += (1.0 - acc.a) * lay.a;
            layPending = false;
            tEnd = t1;
            last = false;
            if (acc.a > 0.985) break;
          }
          t += stepMM;
        }
      }
    }
  }
  if (layPending && acc.a <= 0.985) { // behind the volume (or no volume): over what lies behind
    acc.rgb += (1.0 - acc.a) * lay.rgb;
    acc.a += (1.0 - acc.a) * lay.a;
  }
  float x = md < 1.0 ? uXray * mc.a : 0.0;
  vec3 col = acc.rgb * (1.0 - x) + (1.0 - acc.a * (1.0 - x)) * behind;
  outColor = vec4(pow(max(col, 0.0), vec3(1.0 / 2.2)), 1.0);
}`

function compile(gl, vs, fs) {
  const prog = gl.createProgram()
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type)
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`3D view shader: ${gl.getShaderInfoLog(sh)}`)
    gl.attachShader(prog, sh)
  }
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`3D view shader: ${gl.getProgramInfoLog(prog)}`)
  const u = {}
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS)
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, '')
    u[name] = gl.getUniformLocation(prog, name)
  }
  return { prog, u }
}

// ----------------------------------------------------------------------------------- renderer
export class Render3D {
  constructor(canvas) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true, premultipliedAlpha: false })
    if (!gl) throw new Error('WebGL2 is not available')
    this.gl = gl
    this.meshProg = compile(gl, MESH_VS, MESH_FS)
    this.compProg = compile(gl, QUAD_VS, COMPOSITE_FS)
    this.vao = gl.createVertexArray()
    this.meshes = new Map() // key -> { vao, buffers, count, color, opacity, visible, center, radius }
    this.vol = null // { tex, dims, world2tex, gradToWorld, center, radius }
    this.preset = null
    this.tfTex = null
    this.samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES))
    this.size = [0, 0]
    this.center = [0, 0, 0]
    this.radius = 100
    this.surfaceOpacity = 1
    this.outline = 0.55
    this.xray = 0
    this.resetView()
    this.resize()
  }

  // ---------------------------------------------------------------- camera
  resetView() {
    this.azimuth = 180
    this.elevation = 12
    this.zoom = 1
    this.pan = [0, 0, 0]
    this.draw()
  }

  setView(azimuth, elevation) {
    this.azimuth = azimuth
    this.elevation = Math.min(90, Math.max(-90, elevation))
    this.draw()
  }

  _camera() {
    const az = this.azimuth * DEG, el = this.elevation * DEG
    const back = [-Math.sin(az) * Math.cos(el), -Math.cos(az) * Math.cos(el), Math.sin(el)]
    const right = [Math.cos(az), -Math.sin(az), 0]
    const up = cross(back, right)
    const dist = this.radius / Math.sin(FOV / 2) / this.zoom
    const target = axpy(this.center, 1, this.pan)
    const eye = axpy(target, dist, back)
    const view = new Float32Array([
      right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0,
      -dot(right, eye), -dot(up, eye), -dot(back, eye), 1,
    ])
    const near = Math.max(dist - 1.8 * this.radius, dist * 0.02)
    const far = dist + 1.8 * this.radius
    const [w, h] = this.size
    const proj = perspective(FOV, w / Math.max(h, 1), near, far)
    return { right, up, back, dist, eye, view, proj, near, far, halfH: dist * Math.tan(FOV / 2) }
  }

  /** Zoom by ``factor`` keeping the point under (x, y) (CSS px in the canvas) in place. */
  zoomAt(x, y, factor) {
    const z = Math.min(8, Math.max(0.3, this.zoom * factor))
    const cam = this._camera()
    const r = this.canvas.getBoundingClientRect()
    const nx = (2 * x) / r.width - 1, ny = 1 - (2 * y) / r.height
    const shift = cam.halfH * (1 - this.zoom / z)
    this.pan = axpy(axpy(this.pan, nx * shift * (r.width / r.height), cam.right), ny * shift, cam.up)
    this.zoom = z
    this.draw()
  }

  /** Move the view by (dx, dy) CSS px. */
  panBy(dx, dy) {
    const cam = this._camera()
    const r = this.canvas.getBoundingClientRect()
    const s = (2 * cam.halfH) / Math.max(r.height, 1)
    this.pan = axpy(axpy(this.pan, -dx * s, cam.right), dy * s, cam.up)
    this.draw()
  }

  // ---------------------------------------------------------------- volume
  /** ``data``: intensities in NIfTI voxel order; ``affine``: voxel -> RAS mm (row-major 4x4);
   * ``lo``/``hi``: the intensity range the transfer functions are defined over. */
  setVolume({ dims, data, affine, lo, hi }) {
    const gl = this.gl
    this.clearVolume()
    const [nx, ny, nz] = dims
    const f = new Float32Array(nx * ny * nz)
    const s = 1 / (hi - lo || 1)
    for (let i = 0; i < f.length; i++) f[i] = Math.min(1, Math.max(0, (data[i] - lo) * s))
    const upload = (arr) => {
      const t = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_3D, t)
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16F, nx, ny, nz, 0, gl.RED, gl.FLOAT, arr)
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) {
        gl.texParameteri(gl.TEXTURE_3D, k, v)
      }
      return t
    }
    const tex = upload(f)
    const smooth = upload(blur3d(blur3d(f, dims), dims))
    // world (mm) -> texture coordinates: tex = (inverse(affine) * world + 0.5) / dims
    const A = affine
    const vox2world = new Float32Array([A[0], A[4], A[8], 0, A[1], A[5], A[9], 0, A[2], A[6], A[10], 0, A[3], A[7], A[11], 1])
    const scale = new Float32Array([1 / nx, 0, 0, 0, 0, 1 / ny, 0, 0, 0, 0, 1 / nz, 0, 0.5 / nx, 0.5 / ny, 0.5 / nz, 1])
    const world2tex = mul(scale, invert(vox2world))
    const m = world2tex
    // gradients: d/dworld = transpose(d tex / d world) * d/dtex
    const gradToWorld = new Float32Array([m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]])
    const corners = []
    for (const i of [-0.5, nx - 0.5]) for (const j of [-0.5, ny - 0.5]) for (const k of [-0.5, nz - 0.5]) {
      corners.push([0, 1, 2].map((r) => A[4 * r] * i + A[4 * r + 1] * j + A[4 * r + 2] * k + A[4 * r + 3]))
    }
    const lo3 = [0, 1, 2].map((a) => Math.min(...corners.map((c) => c[a])))
    const hi3 = [0, 1, 2].map((a) => Math.max(...corners.map((c) => c[a])))
    const voxel = Math.cbrt(Math.abs(A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8])))
    this.vol = { tex, smooth, dims, world2tex, gradToWorld, voxel, lo: lo3, hi: hi3 }
    this._reframe(false)
  }

  clearVolume() {
    if (this.vol) for (const t of [this.vol.tex, this.vol.smooth]) this.gl.deleteTexture(t)
    this.vol = null
  }

  /** ``preset``: null (no volume) or { mode: 'shaded' | 'mip', tf: Float32Array(N * 4) colour + opacity,
   * material: Float32Array(N * 4) (x shading), refLen, light: [ambient, diffuse, specular,
   * shininess], grad: [lo, hi], glass: { rgb, alpha, iso, air } } with N table entries over the
   * volume's 0..1 range. Glass: a see-through skin (faces clear, edges opaque) where a ray that
   * starts below ``air`` first reaches ``iso`` (both 0..1). */
  setPreset(preset) {
    const gl = this.gl
    this.preset = preset
    if (preset) {
      const n = preset.tf.length / 4
      // running integrals (trapezoid rule) of extinction and colour x extinction over intensity
      const pre = new Float32Array(n * 4)
      let K = [0, 0, 0], T = 0, prev = null
      for (let i = 0; i < n; i++) {
        const tau = -Math.log(1 - Math.min(preset.tf[4 * i + 3], 0.9999))
        const c = [0, 1, 2].map((k) => Math.pow(preset.tf[4 * i + k], 2.2) * tau)
        if (prev) {
          const h = 0.5 / (n - 1)
          T += h * (prev.tau + tau)
          K = K.map((v, k) => v + h * (prev.c[k] + c[k]))
        }
        prev = { tau, c }
        pre.set([...K, T], 4 * i)
      }
      if (!this.preTex) this.preTex = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, this.preTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, 1, 0, gl.RGBA, gl.FLOAT, pre)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      this.preN = n
      const tab = new Float32Array(n * 8)
      tab.set(preset.tf, 0)
      tab.set(preset.material, n * 4)
      if (!this.tfTex) this.tfTex = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, this.tfTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, n, 2, 0, gl.RGBA, gl.FLOAT, tab)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    }
    this._reframe()
  }

  // The camera frames the volume, or the visible surfaces while no volume is shown (moving there
  // smoothly, so surfaces that come and go do not make the view jump).
  _reframe(animate = true) {
    let lo, hi
    const shown = [...this.meshes.values()].filter((m) => m.visible)
    if ((!this.preset || !this.vol) && shown.length) {
      lo = [0, 1, 2].map((a) => Math.min(...shown.map((m) => m.lo[a])))
      hi = [0, 1, 2].map((a) => Math.max(...shown.map((m) => m.hi[a])))
    } else if (this.vol) {
      ({ lo, hi } = this.vol)
    } else {
      return this.draw()
    }
    const center = lo.map((v, a) => (v + hi[a]) / 2)
    const radius = Math.max(10, Math.hypot(...lo.map((v, a) => hi[a] - v)) / 2)
    this.anim = animate && this.framed ? { c0: this.center, r0: this.radius, c1: center, r1: radius, t0: performance.now() } : null
    if (!this.anim) { this.center = center; this.radius = radius }
    this.framed = true
    this.draw()
  }

  // ---------------------------------------------------------------- meshes
  /** ``pos``: Float32Array of RAS mm vertices, ``tris``: Uint32Array, ``rgb``: 0..255. */
  setMesh(key, pos, tris, rgb, opacity = 1, visible = true) {
    const gl = this.gl
    this.removeMesh(key, false)
    const normals = vertexNormals(pos, tris)
    const vao = gl.createVertexArray()
    gl.bindVertexArray(vao)
    const buffers = []
    for (const [loc, arr] of [[0, pos], [1, normals]]) {
      const b = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, b)
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW)
      gl.enableVertexAttribArray(loc)
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0)
      buffers.push(b)
    }
    const ib = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, tris, gl.STATIC_DRAW)
    buffers.push(ib)
    gl.bindVertexArray(null)
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < pos.length; i += 3) {
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], pos[i + a]); hi[a] = Math.max(hi[a], pos[i + a]) }
    }
    const center = lo.map((v, a) => (v + hi[a]) / 2)
    this.meshes.set(key, {
      vao, buffers, count: tris.length, color: srgbToLinear(rgb.map((c) => c / 255)), opacity, visible,
      lo, hi, center, radius: Math.hypot(...lo.map((v, a) => hi[a] - v)) / 2,
    })
    this._reframe()
  }

  removeMesh(key, redraw = true) {
    const m = this.meshes.get(key)
    if (!m) return
    const gl = this.gl
    gl.deleteVertexArray(m.vao)
    for (const b of m.buffers) gl.deleteBuffer(b)
    this.meshes.delete(key)
    if (redraw) this._reframe()
  }

  setMeshVisible(key, visible) {
    const m = this.meshes.get(key)
    if (m) { m.visible = visible; this._reframe() }
  }

  setMeshOpacity(key, opacity) {
    const m = this.meshes.get(key)
    if (m) { m.opacity = opacity; this.draw() }
  }

  clearMeshes() {
    for (const key of [...this.meshes.keys()]) this.removeMesh(key, false)
    this.framed = false
    this._reframe(false)
  }

  // ---------------------------------------------------------------- frame
  resize() {
    // Ray casting costs per pixel: at most ~1.6 million of them (a high-DPI laptop screen would
    // otherwise quadruple the work of the volume rendering).
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight
    const dpr = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(MAX_PIXELS / Math.max(1, cw * ch)))
    const w = Math.max(1, Math.round(cw * dpr))
    const h = Math.max(1, Math.round(ch * dpr))
    if (w === this.size[0] && h === this.size[1]) return
    this.canvas.width = w
    this.canvas.height = h
    this.size = [w, h]
    this._allocTargets()
    this.draw()
  }

  _allocTargets() {
    const gl = this.gl
    const [w, h] = this.size
    for (const o of this.targets?.renderbuffers || []) gl.deleteRenderbuffer(o)
    for (const o of this.targets?.textures || []) gl.deleteTexture(o)
    for (const o of this.targets?.framebuffers || []) gl.deleteFramebuffer(o)
    const rb = (fmt) => {
      const r = gl.createRenderbuffer()
      gl.bindRenderbuffer(gl.RENDERBUFFER, r)
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, fmt, w, h)
      return r
    }
    const msColor = rb(gl.RGBA8), msDepth = rb(gl.DEPTH_COMPONENT24)
    const ms = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, ms)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, msColor)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, msDepth)
    const tex = (ifmt, fmt, type) => {
      const t = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      return t
    }
    const fbo = (c, d) => {
      const f = gl.createFramebuffer()
      gl.bindFramebuffer(gl.FRAMEBUFFER, f)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, c, 0)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, d, 0)
      return f
    }
    const color = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE)
    const depth = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT)
    const resolved = fbo(color, depth)
    // the semi-transparent surfaces: their own colour, the opaque surfaces' depth buffer
    const msLayerColor = rb(gl.RGBA8)
    const msLayer = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, msLayer)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, msLayerColor)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, msDepth)
    const layerColor = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE)
    const layerDepth = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT)
    const layerResolved = fbo(layerColor, layerDepth)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    this.targets = {
      ms, resolved, color, depth, msLayer, layerResolved, layerColor, layerDepth,
      renderbuffers: [msColor, msDepth, msLayerColor], textures: [color, depth, layerColor, layerDepth],
      framebuffers: [ms, resolved, msLayer, layerResolved],
    }
  }

  /** Redraw at the next animation frame (calls within one frame are merged). */
  draw() {
    if (this._frame) return
    this._frame = requestAnimationFrame(() => {
      this._frame = null
      this.render()
    })
  }

  render() {
    const gl = this.gl
    const [w, h] = this.size
    if (!this.targets) return
    performance.mark('nni-3d-start')
    if (this.anim) {
      const { c0, r0, c1, r1, t0 } = this.anim
      const f = Math.min(1, (performance.now() - t0) / 450)
      const e = f * f * (3 - 2 * f)
      this.center = c0.map((v, a) => v + (c1[a] - v) * e)
      this.radius = r0 + (r1 - r0) * e
      if (f < 1) this.draw()
      else this.anim = null
    }
    const cam = this._camera()
    const viewProj = mul(cam.proj, cam.view)
    const opacityOf = (k, m) => (k === 'current' ? Math.min(0.85, this.surfaceOpacity) : this.surfaceOpacity) * m.opacity
    const visible = [...this.meshes].filter(([, m]) => m.visible && m.count > 0)
    const opaque = visible.filter(([k, m]) => opacityOf(k, m) >= 0.99)
    const transparent = visible.filter(([k, m]) => opacityOf(k, m) < 0.99)

    // 1. opaque surfaces -> multisampled target -> resolved colour + depth textures
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.targets.ms)
    gl.viewport(0, 0, w, h)
    gl.clearColor(0, 0, 0, 0)
    gl.clearDepth(1)
    gl.depthMask(true)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    if (opaque.length) {
      gl.enable(gl.DEPTH_TEST)
      gl.depthFunc(gl.LESS)
      gl.enable(gl.CULL_FACE)
      gl.cullFace(gl.BACK)
      gl.disable(gl.BLEND)
      this._drawMeshes(opaque, cam, () => 1)
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.targets.ms)
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.targets.resolved)
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST)
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.DEPTH_BUFFER_BIT, gl.NEAREST)

    // 2. semi-transparent surfaces in front of the opaque ones, farthest first, each with a depth
    // pre-pass (no inner layers) -> one premultiplied layer + the depth of its nearest surface
    if (transparent.length) {
      const near = (m) => Math.hypot(...[0, 1, 2].map((a) => m.center[a] - cam.eye[a])) - m.radius
      transparent.sort((a, b) => near(b[1]) - near(a[1]))
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.targets.msLayer)
      gl.clear(gl.COLOR_BUFFER_BIT) // the depth buffer keeps the opaque surfaces
      gl.enable(gl.DEPTH_TEST)
      gl.depthFunc(gl.LEQUAL)
      gl.enable(gl.CULL_FACE)
      gl.cullFace(gl.BACK)
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
      for (const item of transparent) {
        gl.colorMask(false, false, false, false)
        gl.depthMask(true)
        gl.disable(gl.BLEND)
        this._drawMeshes([item], cam, () => 1)
        gl.colorMask(true, true, true, true)
        gl.depthMask(false)
        gl.enable(gl.BLEND)
        this._drawMeshes([item], cam, opacityOf)
      }
      gl.depthMask(true)
      gl.disable(gl.BLEND)
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.targets.msLayer)
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.targets.layerResolved)
      gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST)
      gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.DEPTH_BUFFER_BIT, gl.NEAREST)
    }

    // 3. volume over surfaces over background, to the canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, w, h)
    gl.disable(gl.DEPTH_TEST)
    gl.disable(gl.BLEND)
    gl.disable(gl.CULL_FACE)
    const { prog, u } = this.compProg
    gl.useProgram(prog)
    const vol = this.vol, pr = this.preset
    const mode = vol && pr ? (pr.mode === 'mip' ? 2 : 1) : 0
    gl.uniformMatrix4fv(u.uInvViewProj, false, invert(viewProj))
    gl.uniform3fv(u.uEye, cam.eye)
    gl.uniform1i(u.uMode, mode)
    gl.uniform2f(u.uViewport, w, h)
    gl.uniform2f(u.uNearFar, cam.near, cam.far)
    gl.uniform3fv(u.uBgTop, srgbToLinear(BG_TOP))
    gl.uniform3fv(u.uBgBottom, srgbToLinear(BG_BOTTOM))
    gl.uniform1f(u.uOutline, this.outline)
    gl.uniform1f(u.uXray, this.xray)
    const toWorld = (v) => [0, 1, 2].map((a) => cam.right[a] * v[0] + cam.up[a] * v[1] + cam.back[a] * v[2])
    gl.uniform3fv(u.uKeyW, toWorld(KEY))
    gl.uniform3fv(u.uFillW, toWorld(FILL))
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.targets.color)
    gl.uniform1i(u.uMeshColor, 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.targets.depth)
    gl.uniform1i(u.uMeshDepth, 1)
    gl.activeTexture(gl.TEXTURE6)
    gl.bindTexture(gl.TEXTURE_2D, this.targets.layerColor)
    gl.uniform1i(u.uLayerColor, 6)
    gl.activeTexture(gl.TEXTURE7)
    gl.bindTexture(gl.TEXTURE_2D, this.targets.layerDepth)
    gl.uniform1i(u.uLayerDepth, 7)
    gl.uniform1i(u.uLayer, transparent.length ? 1 : 0)
    if (mode) {
      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_3D, vol.tex)
      gl.uniform1i(u.uVol, 2)
      gl.activeTexture(gl.TEXTURE3)
      gl.bindTexture(gl.TEXTURE_2D, this.tfTex)
      gl.uniform1i(u.uTF, 3)
      gl.activeTexture(gl.TEXTURE4)
      gl.bindTexture(gl.TEXTURE_3D, vol.smooth)
      gl.uniform1i(u.uSmooth, 4)
      gl.activeTexture(gl.TEXTURE5)
      gl.bindTexture(gl.TEXTURE_2D, this.preTex)
      gl.uniform1i(u.uPre, 5)
      gl.uniform1f(u.uPreN, this.preN)
      gl.uniformMatrix4fv(u.uWorld2Tex, false, vol.world2tex)
      gl.uniformMatrix3fv(u.uGradToWorld, false, vol.gradToWorld)
      gl.uniform3f(u.uTexel, 1 / vol.dims[0], 1 / vol.dims[1], 1 / vol.dims[2])
      gl.uniform1f(u.uStep, pr.step ?? 0.5)
      gl.uniform1f(u.uRefLen, pr.refLen ?? vol.voxel)
      gl.uniform4fv(u.uLight, pr.light ?? [0.3, 0.75, 0.3, 40])
      gl.uniform2fv(u.uGrad, pr.grad ?? [0.002, 0.02])
      const gs = pr.glass
      gl.uniform4fv(u.uGlass, gs ? [...srgbToLinear(gs.rgb), gs.alpha] : [0, 0, 0, 0])
      gl.uniform2fv(u.uGlassIso, gs ? [gs.iso, gs.air] : [2, -1])
    } else {
      // samplers must not be left pointing at the wrong texture types
      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_3D, this._emptyTex3d())
      gl.uniform1i(u.uVol, 2)
      gl.activeTexture(gl.TEXTURE3)
      gl.bindTexture(gl.TEXTURE_2D, this._emptyTex2d())
      gl.uniform1i(u.uTF, 3)
      gl.activeTexture(gl.TEXTURE4)
      gl.bindTexture(gl.TEXTURE_3D, this._emptyTex3d())
      gl.uniform1i(u.uSmooth, 4)
      gl.activeTexture(gl.TEXTURE5)
      gl.bindTexture(gl.TEXTURE_2D, this._emptyTex2d())
      gl.uniform1i(u.uPre, 5)
    }
    gl.bindVertexArray(this.vao)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
    gl.bindVertexArray(null)
    performance.measure('nni-3d', 'nni-3d-start')
  }

  _drawMeshes(items, cam, opacityOf) {
    const gl = this.gl
    const { prog, u } = this.meshProg
    gl.useProgram(prog)
    gl.uniformMatrix4fv(u.uView, false, cam.view)
    gl.uniformMatrix4fv(u.uProj, false, cam.proj)
    gl.uniform3fv(u.uKey, KEY)
    gl.uniform3fv(u.uFill, FILL)
    for (const [key, m] of items) {
      gl.uniform3fv(u.uColor, m.color)
      gl.uniform1f(u.uOpacity, opacityOf(key, m))
      gl.bindVertexArray(m.vao)
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0)
    }
  }

  _emptyTex3d() {
    if (!this._e3) {
      const gl = this.gl
      this._e3 = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_3D, this._e3)
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16F, 1, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1))
    }
    return this._e3
  }

  _emptyTex2d() {
    if (!this._e2) {
      const gl = this.gl
      this._e2 = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, this._e2)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4))
    }
    return this._e2
  }
}
