import mongoose from 'mongoose';
const { Schema, model, Model } = mongoose;

export interface IUser {
  role: string;
  uuid: string;
  public_key: string;
}

export const UserSchema = new Schema<IUser>({
  role: String,
  uuid: String,
  public_key: String,
});

export const User = model<IUser>('User', UserSchema);
