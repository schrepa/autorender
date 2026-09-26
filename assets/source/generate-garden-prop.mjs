// Repository-owned procedural source art. No external assets or npm packages.
// Run: node assets/source/generate-garden-prop.mjs
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const name = Buffer.from(type);
  const length = Buffer.alloc(4); length.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([name, data])));
  return Buffer.concat([length, name, data, crc]);
}
const pixels = Buffer.alloc(64 * (64 * 4 + 1));
for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
  const u = x % 32, v = y % 32;
  const noise = ((x * 13 + y * 29 + x * y * 3) % 11) - 5;
  let rgb;
  if (x < 32 && y < 32) { // warm stone blocks with narrow mortar
    const seam = v % 8 === 0 || (u + (Math.floor(v / 8) % 2) * 8) % 16 === 0;
    rgb = seam ? [116, 128, 132] : [188 + noise, 195 + noise, 184 + noise];
  } else if (x >= 32 && y < 32) { // copper-brown carved wood, grain and joints
    const grain = (u * 3 + Math.floor(Math.sin(v / 5) * 2)) % 9;
    rgb = u % 8 === 0 ? [100, 55, 30] : [179 + noise - grain * 3, 114 + noise - grain * 2, 56 + noise];
  } else if (x < 32) { // dense decorative moss with little golden flowers
    const flower = (u * 7 + v * 11) % 47 < 2;
    rgb = flower ? [229, 194, 86] : [53 + noise, 120 + noise + (u + v) % 9, 65 + noise];
  } else { // teal glazed roof tiles and gold inset lines
    const seam = u % 8 === 0 || v % 8 === 0;
    const inlay = (u + v) % 16 === 0;
    rgb = inlay ? [227, 184, 75] : seam ? [26, 67, 78] : [48 + noise, 126 + noise, 140 + noise];
  }
  const offset = y * 257 + 1 + x * 4;
  pixels.set([...rgb, 255], offset);
}
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(64); ihdr.writeUInt32BE(64, 4); ihdr[8] = 8; ihdr[9] = 6;
const png = Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), pngChunk('IHDR', ihdr), pngChunk('IDAT', deflateSync(pixels, { level: 9 })), pngChunk('IEND', Buffer.alloc(0))]);

const groups = [[], []];
function face(points, tile, group = 0) {
  const a = points[1].map((value, i) => value - points[0][i]);
  const b = points[2].map((value, i) => value - points[0][i]);
  const cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const length = Math.hypot(...cross), normal = cross.map(value => value / length);
  const tx = tile % 2 * 32, ty = Math.floor(tile / 2) * 32;
  const uv = [[0.5,31.5],[31.5,31.5],[31.5,0.5],[0.5,0.5]];
  groups[group].push({ vertices: points.map((position, i) => ({ position, normal, uv: [(tx + uv[i][0]) / 64, (ty + uv[i][1]) / 64] })), indices: points.length === 4 ? [0,1,2,0,2,3] : [0,1,2] });
}
function box(x, y, z, width, height, depth, tile, group = 0) {
  const l=x-width/2, r=x+width/2, b=y, t=y+height, n=z-depth/2, f=z+depth/2;
  face([[l,b,f],[r,b,f],[r,t,f],[l,t,f]], tile, group);
  face([[r,b,n],[l,b,n],[l,t,n],[r,t,n]], tile, group);
  face([[r,b,f],[r,b,n],[r,t,n],[r,t,f]], tile, group);
  face([[l,b,n],[l,b,f],[l,t,f],[l,t,n]], tile, group);
  face([[l,t,f],[r,t,f],[r,t,n],[l,t,n]], tile, group);
  face([[l,b,n],[r,b,n],[r,b,f],[l,b,f]], tile, group);
}
box(0, 0, 0, 3.2, 0.35, 3.2, 0);
box(0, 0.35, 0, 2.85, 0.55, 2.85, 0);
box(0, 0.9, 0, 2.5, 0.08, 2.5, 2);
box(0, 0.98, 0, 0.48, 1.35, 0.48, 1, 1);
box(0, 2.15, 0, 1.25, 0.72, 1.25, 1, 1);
box(0, 2.87, 0, 1.7, 0.12, 1.7, 3);
const top = [0,3.65,0], low = 2.99, extent = 1.04;
face([[-extent,low,extent],[extent,low,extent],top], 3);
face([[extent,low,extent],[extent,low,-extent],top], 3);
face([[extent,low,-extent],[-extent,low,-extent],top], 3);
face([[-extent,low,-extent],[-extent,low,extent],top], 3);

const binaryParts = [], bufferViews = [], accessors = [];
let binaryLength = 0;
function view(bytes, target) {
  while (binaryLength % 4) { binaryParts.push(Buffer.alloc(1)); binaryLength++; }
  const index = bufferViews.length;
  bufferViews.push({ buffer: 0, byteOffset: binaryLength, byteLength: bytes.length, ...(target ? { target } : {}) });
  binaryParts.push(bytes); binaryLength += bytes.length;
  return index;
}
function accessor(values, type, componentType, dimensions, bounds = false) {
  const bytes = Buffer.alloc(values.length * (componentType === 5126 ? 4 : 2));
  values.forEach((value, index) => componentType === 5126 ? bytes.writeFloatLE(value, index * 4) : bytes.writeUInt16LE(value, index * 2));
  const index = accessors.length;
  const result = { bufferView: view(bytes, componentType === 5126 ? 34962 : 34963), componentType, count: values.length / dimensions, type };
  if (bounds) {
    result.min = Array.from({length:dimensions}, (_, axis) => Math.min(...values.filter((_, i) => i % dimensions === axis)));
    result.max = Array.from({length:dimensions}, (_, axis) => Math.max(...values.filter((_, i) => i % dimensions === axis)));
  }
  accessors.push(result); return index;
}
const primitives = groups.map((faces, material) => {
  const vertices = [], indices = [];
  for (const current of faces) { indices.push(...current.indices.map(index => index + vertices.length)); vertices.push(...current.vertices); }
  return { attributes: {
    POSITION: accessor(vertices.flatMap(vertex => vertex.position), 'VEC3', 5126, 3, true),
    NORMAL: accessor(vertices.flatMap(vertex => vertex.normal), 'VEC3', 5126, 3),
    TEXCOORD_0: accessor(vertices.flatMap(vertex => vertex.uv), 'VEC2', 5126, 2),
  }, indices: accessor(indices, 'SCALAR', 5123, 1), material, mode: 4 };
});
const imageView = view(png);
const json = {
  asset: { version:'2.0', generator:'Browser Engine procedural garden source v1', copyright:'Original repository-owned procedural geometry and texture; no third-party source assets' },
  scene:0, scenes:[{nodes:[0]}], nodes:[{name:'Garden shrine',children:[1]},{name:'Garden shrine stone and carved wood',mesh:0}], meshes:[{primitives}],
  materials:[0,1].map(index => ({ name:index ? 'Carved wood' : 'Stone moss and glazed roof', pbrMetallicRoughness:{baseColorFactor:[1,1,1,1],baseColorTexture:{index:0},metallicFactor:0,roughnessFactor:1} })),
  textures:[{sampler:0,source:0}], samplers:[{magFilter:9729,minFilter:9729,wrapS:33071,wrapT:33071}], images:[{bufferView:imageView,mimeType:'image/png'}],
  buffers:[{byteLength:binaryLength}], bufferViews, accessors,
};
let jsonBytes = Buffer.from(JSON.stringify(json)); while (jsonBytes.length % 4) jsonBytes = Buffer.concat([jsonBytes, Buffer.from(' ')]);
let binary = Buffer.concat(binaryParts); while (binary.length % 4) binary = Buffer.concat([binary, Buffer.alloc(1)]);
const header=Buffer.alloc(12); header.write('glTF'); header.writeUInt32LE(2,4); header.writeUInt32LE(28 + jsonBytes.length + binary.length,8);
const jsonHeader=Buffer.alloc(8); jsonHeader.writeUInt32LE(jsonBytes.length); jsonHeader.write('JSON',4);
const binHeader=Buffer.alloc(8); binHeader.writeUInt32LE(binary.length); binHeader.write('BIN\0',4);
const output = fileURLToPath(new URL('garden-prop.glb', import.meta.url));
writeFileSync(output, Buffer.concat([header,jsonHeader,jsonBytes,binHeader,binary]));
console.log(JSON.stringify({output,bytes:28+jsonBytes.length+binary.length,vertices:groups.flat().reduce((sum, item)=>sum+item.vertices.length,0),texture:'64x64'}));
