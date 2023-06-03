import mongoose from 'mongoose';
const { Schema, model, Model } = mongoose;

export interface INews {
  title: string;
  url: string;
  date: Date;
  authors: string[];
  content: string;
  categories: string[];
  views: Date[];
  location: {
    type: 'Point';
    coordinates: [number, number];
  };
}

export const NewsSchema = new Schema<INews>({
  title: String,
  url: String,
  date: Date,
  authors: [String],
  content: String,
  categories: [String],
  views: [Date],
  location: {
    type: {
      type: String,
      enum: ['Point'],
      required: true,
    },
    coordinates: {
      type: [Number],
      required: true,
    },
  },
});

export const News = model<INews>('News', NewsSchema);
