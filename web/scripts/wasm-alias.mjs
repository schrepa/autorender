// @ts-check
export const wasmImport = (name, suffix = "") => new RegExp(`^(?:\\.\\./)+pkg/${name}${suffix}$`);
