// `textureNumLayers` of a storage texture queries its size without a level of detail.
@group(0) @binding(0) var s_array_w: texture_storage_2d_array<r32uint, write>;
@compute @workgroup_size(1) fn main() {
    let layers: u32 = textureNumLayers(s_array_w);
    textureStore(s_array_w, vec2u(0), 0, vec4u(layers));
}
