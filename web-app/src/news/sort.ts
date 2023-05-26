import INews from "./model";

export function dateAsc(news: INews[]): INews[] {
  return news.sort((a, b) => a.date.getTime() - b.date.getTime());
}

export function dateDesc(news: INews[]): INews[] {
  return news.sort((a, b) => b.date.getTime() - a.date.getTime());
}
