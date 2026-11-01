import package::face::exposure::adapted;
import package::face::sky::adapted_exposure;
import package::generated::pieces::tonemap::piece;

fn exposure() -> f32 {
    return adapted(adapted_exposure(piece().exposure.x));
}
