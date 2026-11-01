import package::place::sun_blockers::{SunAt, SunOpen};
import package::sky::faces::sky::cloud_passing;

fn sun_open(open: SunOpen, at: SunAt) -> SunOpen {
    var result = open;
    result.passing = open.passing * cloud_passing(at.world);
    return result;
}
