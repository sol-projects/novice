import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
import { closeServer } from '../index';

import { INews, News } from '../model/News';

let errorOccurred = false;

afterEach(() => {
  if (errorOccurred) {
    console.error('Error occurred in one of the tests');
    errorOccurred = false;
  }
});

test('GET /news - should return all news', async () => {
  try {
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
  } catch (error) {
    console.error(error);
    errorOccurred = true;
  }
}, 50000);

test('GET /news - categories should be an array of strings', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(Array.isArray(newsItem.categories)).toBeTruthy();
    newsItem.categories.forEach((category: string) => {
      expect(typeof category).toBe('string');
    });
  });
});

test('GET /news - each title should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.title).toBe('string');
  });
});

test('GET /news - each author should be an array of strings', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(Array.isArray(newsItem.authors)).toBeTruthy();
    newsItem.authors.forEach((author: string) => {
      expect(typeof author).toBe('string');
    });
  });
});

test('GET /news - each content should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.content).toBe('string');
  });
});

test('GET /news - each location should be an object with the expected properties', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.location).toBe('object');
    expect(newsItem.location).toHaveProperty('type');
    expect(newsItem.location).toHaveProperty('coordinates');
    expect(newsItem.location.type).toBe('Point');
    expect(Array.isArray(newsItem.location.coordinates)).toBeTruthy();
    expect(newsItem.location.coordinates.length).toBe(2);
    expect(typeof newsItem.location.coordinates[0]).toBe('number');
    expect(typeof newsItem.location.coordinates[1]).toBe('number');
  });
});

test('GET /news - should return an array of news items', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();
});

import puppeteer from 'puppeteer';

test('Maribor info 5 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/mbinfo/1`);

  // Wait for the page to load and get the JSON response
  const jsonResponse = await page.evaluate(() => {
    const preElement = document.querySelector('pre');
    return JSON.parse(preElement!.textContent!);
  });

  await browser.close();

  // Extract the news items from the JSON response
  const newsItems = jsonResponse.map((item: any) => ({
    title: item.title,
    url: item.url,
  }));

  // Assertions
  expect(newsItems).toHaveLength(1);
}, 200000);

test('Siol 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/siol/1`);

  // Wait for the page to load and get the JSON response
  const jsonResponse = await page.evaluate(() => {
    const preElement = document.querySelector('pre');
    return JSON.parse(preElement!.textContent!);
  });

  await browser.close();

  // Extract the news items from the JSON response
  const newsItems = jsonResponse.map((item: any) => ({
    title: item.title,
    url: item.url,
  }));

  // Assertions
  expect(newsItems).toHaveLength(1);
}, 2000000);

test('Sta 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/sta/1`);

  // Wait for the page to load and get the JSON response
  const jsonResponse = await page.evaluate(() => {
    const preElement = document.querySelector('pre');
    return JSON.parse(preElement!.textContent!);
  });

  await browser.close();

  // Extract the news items from the JSON response
  const newsItems = jsonResponse.map((item: any) => ({
    title: item.title,
    url: item.url,
  }));

  // Assertions
  expect(newsItems).toHaveLength(1);
  await mongoose.disconnect();
  closeServer(); // Close the HTTP server
}, 200000);

afterAll(async () => {
  try {
    await mongoose.disconnect(); // Close the database connection
  } catch (error) {
    console.error(error);
  }
});
