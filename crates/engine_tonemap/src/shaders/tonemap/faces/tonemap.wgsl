import package::face::exposure::adapted;
import package::generated::pieces::tonemap::piece;

fn exposure() -> f32 {
    return adapted(piece().exposure.x);
}
