import mongoose from 'mongoose';
const { Schema, model } = mongoose;

const news_schema = new Schema({
  title: String,
  date: Date,
  author: String,
  content: String,
  image_info: String,
  categories: [String],
  location: String,
});

const news_schema = model('news_schema', blogSchema);
export default news_schema;
