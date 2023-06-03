import INews from "./model";

export async function getAll() {
  try {
    const response = await fetch("http://localhost:8000/news");
    if (!response.ok) {
      console.log("Failed to fetch news");
    }

    const data = await response.json();
    const news: INews[] = data.map((item: any) => {
      const date = new Date(item.date);
      const views = item.views.map((view: any) => new Date(view));
      return { ...item, date, views };
    });

    news.sort(function (a, b) {
      return b.date.getTime() - a.date.getTime();
    });

    return news;
  } catch (error) {
    console.error("Error fetching news:", error);
  }

  return undefined;
}
