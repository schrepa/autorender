import package::generated::kinds::KIND_PLACED;
import package::generated::scatter::PlacedValues;

@group(0) @binding(0) var<storage, read> placed_values: array<PlacedValues>;

fn placed_values_of(kind: u32, object: u32) -> PlacedValues {
    var values: PlacedValues;
    if kind != KIND_PLACED || object >= arrayLength(&placed_values) {
        return values;
    }
    return placed_values[object];
}
