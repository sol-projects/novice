import mongoose from 'mongoose';
const { Schema, model, connect, Model } = mongoose;

export interface INews {
  title: string;
  url: string;
  date: Date;
  author: string;
  content: string;
  image_info: string;
  categories: string[];
  location: string;
}

export const NewsSchema = new Schema<INews>({
  title: String,
  url: String,
  date: Date,
  author: String,
  content: String,
  image_info: String,
  categories: [String],
  location: String,
});

export const News = model<INews>('News', NewsSchema);
