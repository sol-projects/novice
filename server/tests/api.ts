import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
import { INews, News } from '../model/News';
import dotenv from 'dotenv';
import * as Db from '../db/db'

Db.disconnect();
if(process.env.DB_NAME_TEST ) {
    Db.connect(process.env.DB_NAME_TEST);
}

test('login', async () => {
  const res = await request(app).post('/login');
  expect(res.headers['content-type']).toMatch('/json/');
  console.log(res.body);
});


