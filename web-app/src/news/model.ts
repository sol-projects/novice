export default interface INews {
  _id: string;
  title: string;
  url: string;
  date: Date;
  authors: string[];
  content: string;
  categories: string[];
  location: {
    type: "Point";
    coordinates: [number, number];
  };
}
