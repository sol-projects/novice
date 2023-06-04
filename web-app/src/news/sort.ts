import INews from "./model";

export function dateAsc(news: INews[]): INews[] {
  return news.sort((a, b) => a.date.getTime() - b.date.getTime());
}

export function dateDesc(news: INews[]): INews[] {
  return news.sort((a, b) => b.date.getTime() - a.date.getTime());
}

export function views(news: INews[]): INews[] {
  return news.sort((a, b) => b.views.length - a.views.length);
}

export function popularity(news: INews[]): INews[] {
  const currentTime = new Date();

  return news.sort((a, b) => {
    const aWeightedViews = calculateWeightedViews(a, currentTime);
    const bWeightedViews = calculateWeightedViews(b, currentTime);

    return bWeightedViews - aWeightedViews;
  });
}

function calculateWeightedViews(newsItem: INews, currentTime: Date): number {
  let weightedViews = newsItem.views.length;

  for (const viewTime of newsItem.views) {
    const timeDifference = currentTime.getTime() - viewTime.getTime();
    const weight = calculateWeight(timeDifference);
    weightedViews += weight;
  }

  return weightedViews;
}

function calculateWeight(timeDifference: number): number {
  const timeThresholds = [
    { threshold: 60 * 60 * 1000, weight: 1 },
    { threshold: 6 * 60 * 60 * 1000, weight: 0.8 },
    { threshold: 24 * 60 * 60 * 1000, weight: 0.5 },
    { threshold: 2 * 24 * 60 * 60 * 1000, weight: 0.05 },
    { threshold: 7 * 24 * 60 * 60 * 1000, weight: 0.001 },
  ];

  for (const timeThreshold of timeThresholds) {
    if (timeDifference <= timeThreshold.threshold) {
      return timeThreshold.weight * 10;
    }
  }

  return 0;
}
