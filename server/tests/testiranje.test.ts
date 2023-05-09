import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';

import { INews, News } from '../model/News';

test('GET /news - should return all news', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(res.headers['content-type']).toMatch(/json/);
  expect(Array.isArray(res.body)).toBeTruthy();

  // Check if the response data contains the expected properties
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

  // Check if the response data contains the expected properties and valid categories
  res.body.forEach((newsItem: any) => {
    expect(newsItem).toHaveProperty('_id');
    expect(newsItem).toHaveProperty('title');
    expect(newsItem).toHaveProperty('url');
    expect(newsItem).toHaveProperty('date');
    expect(newsItem).toHaveProperty('authors');
    expect(newsItem).toHaveProperty('content');
    expect(newsItem).toHaveProperty('categories');
    expect(newsItem).toHaveProperty('location');

    // Check if categories is an array and has valid values
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
});



