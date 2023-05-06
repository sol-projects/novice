import mongoose from 'mongoose';
const { Schema, model, Model } = mongoose;

export interface INews {
  title: string;
  url: string;
  date: Date;
  authors: string[];
  content: string;
  categories: string[];
  location: string;
}

export const NewsSchema = new Schema<INews>({
  title: String,
  url: String,
  date: Date,
  authors: [String],
  content: String,
  categories: [String],
  location: String,
});

export const News = model<INews>('News', NewsSchema);
