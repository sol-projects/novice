import mongoose from 'mongoose';

export function connect() {
  if (!process.env) {
    console.error(
      'You must call dotenv.config() before calling this function.'
    );
  } else {
    mongoose.connect(
      `mongodb+srv://${process.env.DB_USERNAME}:${process.env.DB_PASSWORD}@cluster0.bd0tfwp.mongodb.net/?retryWrites=true&w=majority`
    );
  }
}

export namespace Util {
  export function toCoords() {
    return [0, 0];
  }
}
