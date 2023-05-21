import INews from "./model";

export async function getAll() {
  try {
    const response = await fetch("http://localhost:8000/news");
    if (!response.ok) {
      console.log("Failed to fetch orders");
    }

    const data = await response.json();
    const news: INews[] = data.map((item: any) => {
      const date = new Date(item.date);
      return { ...item, date };
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
