/** Image `uri`'s RGBA8 pixels, rows from the top, decoded by the browser's
 * own decoder with color and alpha as stored; on the page or in a job
 * worker. `uri` names it in messages. */
export async function decodeImagePixels(uri: string, blob: Blob): Promise<{ width: number; height: number; pixels: Uint8Array }> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error(`Scene file ${uri} could not be decoded.`);
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    return { width: bitmap.width, height: bitmap.height, pixels: new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength) };
  } finally {
    bitmap.close();
  }
}
