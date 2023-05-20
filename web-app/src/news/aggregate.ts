import INews from "./model";

export type Aggregation = {
  key: string;
  value: number;
};

export function byTopCategories(news: INews[], n: number): Aggregation[] {
  const frequencies: Record<string, number> = {};

  for (const newsItem of news) {
    for (const category of newsItem.categories) {
      frequencies[category] = (frequencies[category] || 0) + 1;
    }
  }

  return sort(Object.entries(frequencies)).slice(0, n);
}

export function byTopAuthors(news: INews[], n: number): Aggregation[] {
  const frequencies: Record<string, number> = {};

  for (const article of news) {
    for (const author of article.authors) {
      frequencies[author] = (frequencies[author] || 0) + 1;
    }
  }

  return sort(Object.entries(frequencies)).slice(0, n);
}

export function byDate(news: INews[], n: number): Aggregation[] {
  const frequencies: Record<string, number> = {};

  for (const newsItem of news) {
    const dateKey = newsItem.date.toISOString().split("T")[0];

    frequencies[dateKey] = (frequencies[dateKey] || 0) + 1;
  }

  return sort(Object.entries(frequencies)).slice(0, n);
}

function sort(entries: [string, number][]): Aggregation[] {
  entries.sort((a, b) => {
    return b[1] - a[1];
  });

  return entries.map(([key, value]) => ({ key, value }));
}
