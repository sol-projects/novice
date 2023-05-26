import INews from "./model";

export function date(news: INews[], from: Date, to: Date): INews[] {
  return news.filter((item) => item.date >= from && item.date <= to);
}

export function authors(news: INews[], authors: string[]): INews[] {
  return news.filter((item) =>
    authors.some((author) => item.authors.includes(author))
  );
}

export function categories(news: INews[], categories: string[]): INews[] {
  return news.filter((item) =>
    categories.some((category) => item.categories.includes(category))
  );
}

export function title(news: INews[], title: string): INews[] {
  return news.filter((item) => item.title.includes(title));
}

export function content(news: INews[], content: string): INews[] {
  return news.filter((item) => item.title.includes(content));
}

export function websites(news: INews[], websites: string[]): INews[] {
  return news.filter((item) =>
    websites.some((website) => item.url.includes(website))
  );
}
