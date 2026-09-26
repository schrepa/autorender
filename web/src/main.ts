import { orbitBindings } from "./bindings";
import { boot } from "./host";

boot({
  async create(canvas, count) {
    const { default: init, Game } = await orbitBindings();
    await init();
    return Game.create(canvas, count);
  },
});
