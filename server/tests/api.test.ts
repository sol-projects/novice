import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
import * as Db from '../db/db';
import { INews, News } from '../model/News';
import dotenv from 'dotenv';

dotenv.config();
if (process.env.DB_NAME_TEST) {
  Db.connect(process.env.DB_NAME_TEST);
}

test('login', async () => {
  const res = await request(app).post('/login');
  expect(res.headers['content-type']).toMatch(/json/);
  console.log(res.body);
});

/*test('GET /news - should return all news', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(res.headers['content-type']).toMatch(/json/);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(newsItem).toHaveProperty('_id');
    expect(newsItem).toHaveProperty('title');
    expect(newsItem).toHaveProperty('url');
    expect(newsItem).toHaveProperty('date');
    expect(newsItem).toHaveProperty('authors');
    expect(newsItem).toHaveProperty('content');
    expect(newsItem).toHaveProperty('categories');
    expect(newsItem).toHaveProperty('location');
  });
},50000);

test('GET /news - should get 5 news items and check categories and their values', async () => {
  const res = await request(app).get('/news?limit=5');
  expect(res.status).toEqual(200);
  expect(res.headers['content-type']).toMatch(/json/);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(newsItem).toHaveProperty('_id');
    expect(newsItem).toHaveProperty('title');
    expect(newsItem).toHaveProperty('url');
    expect(newsItem).toHaveProperty('date');
    expect(newsItem).toHaveProperty('authors');
    expect(newsItem).toHaveProperty('content');
    expect(newsItem).toHaveProperty('categories');
    expect(newsItem).toHaveProperty('location');

    expect(Array.isArray(newsItem.categories)).toBeTruthy();
    newsItem.categories.forEach((category: string) => {
      expect(typeof category).toBe('string');
      expect(category.trim()).not.toEqual('');
    });
  });
});

describe('POST and DELETE news', () => {
  let savedNewsId: string;
  const testNews = {
    title: 'Test News',
    content: 'This is a test news article.',
    authors: ['Test Author'],
    categories: ['Test Category'],
    location: 'Test Location',
  };

  test('POST news', async () => {
    const response = await request(app)
      .post('/news')
      .send(testNews)
      .expect(201);

    expect(response.body.title).toEqual(testNews.title);
    expect(response.body.content).toEqual(testNews.content);
    expect(response.body.authors).toEqual(testNews.authors);
    expect(response.body.categories).toEqual(testNews.categories);
    expect(response.body.location).toEqual(testNews.location);
    savedNewsId = response.body._id;
  });

  test('DELETE news', async () => {
    await request(app).delete(`/news/${savedNewsId}`).expect(200);

    const deletedNews = await News.findById(savedNewsId);
    expect(deletedNews).toBeNull();
  });
});*/
