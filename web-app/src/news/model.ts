export default interface INews {
  _id: string;
  title: string;
  url: string;
  date: Date;
  authors: string[];
  content: string;
  categories: string[];
  views: Date[];
  location: {
    type: "Point";
    coordinates: [number, number];
  };
}
