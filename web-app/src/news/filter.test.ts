import { content } from "./filter";
import INews from "./model";

test("content search matches article content rather than its title", () => {
  const article = { title: "Naslov", content: "Besedilo novice" } as INews;
  expect(content([article], "Besedilo")).toEqual([article]);
  expect(content([article], "Naslov")).toEqual([]);
});
