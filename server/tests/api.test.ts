import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
import { INews, News } from '../model/News';
import dotenv from 'dotenv';
import * as Db from '../db/db'
import { closeServer } from '../index';

if(process.env.DB_NAME_TEST ) {
    Db.connect(process.env.DB_NAME_TEST);
}

test('login', async () => {
  const res = await request(app).post('/news/login');
  expect(res.status).toBe(200);
  expect(res.body).toHaveProperty('token');
  let token = res.body.token;
});

afterAll(async () => {
  try {
    await mongoose.disconnect(); // Close the database connection
  } catch (error) {
    console.error(error);
  }
});

let errorOccurred = false;

afterEach(() => {
  if (errorOccurred) {
    console.error('Error occurred in one of the tests');
    errorOccurred = false;
  }
});

// Testing POST request
test('POST /news - should create a new news item', async () => {
  try {
    const newNewsItem: INews = {
      title: 'Test Title',
      url: 'https://testurl.com',
      date: new Date(),
      authors: ['Test Author'],
      content: 'Test content',
      categories: ['Test Category'],
      location: { type: 'Point', coordinates: [0, 0] },
    };
  
    const res = await request(app)
      .post('/news')
      .send(newNewsItem);
      
    expect(res.status).toEqual(201);
    expect(res.headers['content-type']).toMatch(/json/);
    expect(res.body.title).toBe(newNewsItem.title);
    expect(res.body.url).toBe(newNewsItem.url);
  } catch (error) {
    console.error(error);
    errorOccurred = true;
  }
}, 10000);

// Testing PUT (update) request
test('PUT /news/:id - should update a news item', async () => {
  try {
    const updateFields = {
      title: 'Updated Title',
      content: 'Updated content',
    };

    // Assuming we're updating the news item created in the previous test
    const res = await request(app)
      .put('/news/1') // replace '1' with actual ID
      .send(updateFields);
    
    expect(res.status).toEqual(200);
    expect(res.headers['content-type']).toMatch(/json/);
    expect(res.body.title).toBe(updateFields.title);
    expect(res.body.content).toBe(updateFields.content);
  } catch (error) {
    console.error(error);
    errorOccurred = true;
  }
}, 100000);



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
}, 500000);

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
}, 200000);

test('GET /news - each title should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.title).toBe('string');
  });
}, 200000);

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
}, 200000);

test('GET /news - each content should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.content).toBe('string');
  });
}, 200000);

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
}, 200000);

test('GET /news - should return an array of news items', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();
}, 200000);

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
}, 500000);

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
}, 200000);

test('GET /news - each title should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.title).toBe('string');
  });
}, 200000);

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
}, 200000);

test('GET /news - each content should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.content).toBe('string');
  });
}, 200000);

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
}, 200000);

test('GET /news - should return an array of news items', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();
}, 200000);

test('GET /news - each category should be a string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(Array.isArray(newsItem.categories)).toBeTruthy();
    newsItem.categories.forEach((category: string) => {
      expect(typeof category).toBe('string');
    });
  });
}, 200000);

test('GET /news - each news item should have a unique _id', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  const ids = res.body.map((newsItem: any) => newsItem._id);
  const uniqueIds = new Set(ids);

  expect(uniqueIds.size).toBe(ids.length);
}, 200000);

test('GET /news - each news item should have a valid URL', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(newsItem.url).toMatch(/^http(s)?:\/\/[^\s/$.?#].[^\s]*$/);
  });
}, 200000);

test('GET /news - each news item should have a valid date format', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(new Date(newsItem.date)).not.toEqual('Invalid Date');
  });
}, 200000);

test('GET /news - each content should be a non-empty string', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.content).toBe('string');
    expect(newsItem.content.length).toBeGreaterThan(0);
  });
}, 200000);

test('GET /news - each location should have valid coordinates', async () => {
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
}, 200000);

test('GET /news - should return at least one news item', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();
  expect(res.body.length).toBeGreaterThan(0);
}, 200000);



test('GET /news - each news item should have a valid location', async () => {
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
}, 200000);

test('GET /news - each news item should have a non-empty content', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.content).toBe('string');
    expect(newsItem.content.length).toBeGreaterThan(0);
  });
}, 200000);

test('GET /news - each news item should have a valid title length', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.title).toBe('string');
    expect(newsItem.title.length).toBeGreaterThanOrEqual(0);
    expect(newsItem.title.length).toBeLessThanOrEqual(220);
  });
}, 200000);

test('GET /news - each news item should have a valid location type', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(typeof newsItem.location).toBe('object');
    expect(newsItem.location.type).toBe('Point');
  });
}, 200000);

test('GET /news - each news item should have a valid location coordinates range', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    expect(Array.isArray(newsItem.location.coordinates)).toBeTruthy();
    expect(newsItem.location.coordinates.length).toBe(2);
    expect(typeof newsItem.location.coordinates[0]).toBe('number');
    expect(typeof newsItem.location.coordinates[1]).toBe('number');
  });
}, 200000);



test('GET /news - each news item should have a valid ID format', async () => {
  const res = await request(app).get('/news');
  expect(res.status).toEqual(200);
  expect(Array.isArray(res.body)).toBeTruthy();

  res.body.forEach((newsItem: any) => {
    const idRegex = /^[a-fA-F0-9]{24}$/;
    expect(idRegex.test(newsItem._id)).toBeTruthy();
  });
}, 200000);

test('GET /news - each news item should have a valid location type and coordinates format', async () => {
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
    expect(newsItem.location.coordinates[0]).toBeGreaterThanOrEqual(-180);
    expect(newsItem.location.coordinates[0]).toBeLessThanOrEqual(180);
    expect(newsItem.location.coordinates[1]).toBeGreaterThanOrEqual(-180);
    expect(newsItem.location.coordinates[1]).toBeLessThanOrEqual(180);
    closeServer(); // Close the HTTP server

  });
}, 200000);

