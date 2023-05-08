import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
const DB_URI = "mongodb+srv://senad:senad@cluster0.bd0tfwp.mongodb.net/?retryWrites=true&w=majority";

import { INews } from '../model/News';

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

test('GET /news/scrape/:n - should scrape a specific number of websites and return news objects', async () => {
  const numberOfWebsites = 3;
  const res = await request(app).get(`/news/scrape/${numberOfWebsites}`);
  expect(res.status).toEqual(200);
  expect(res.headers['content-type']).toMatch(/json/);
  expect(Array.isArray(res.body)).toBeTruthy();

  // Check if the response data contains the expected properties
  res.body.forEach((newsItem: any) => {
    expect(newsItem).toHaveProperty('title');
    expect(newsItem).toHaveProperty('url');
    expect(newsItem).toHaveProperty('date');
    expect(newsItem).toHaveProperty('authors');
    expect(newsItem).toHaveProperty('content');
    expect(newsItem).toHaveProperty('categories');
    expect(newsItem).toHaveProperty('location');
  });

  // Check if the returned objects are greater than or equal to the number of websites
  expect(res.body.length).toBeGreaterThanOrEqual(numberOfWebsites);
}, 50000); // Set timeout to 50,000ms to accommodate scraping time

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

  // Check if the returned objects are equal to the specified limit
  expect(res.body.length).toEqual(5);
});
