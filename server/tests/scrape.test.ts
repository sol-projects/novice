import mongoose from 'mongoose';
import request from 'supertest';
import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import app from '../index';
import { closeServer } from '../index';
import { INews, News } from '../model/News';

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


test('Delo 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/delo/1`);

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

test('Gov-vlada 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/gov-vlada/1`);

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

test('svet24 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/svet24/1`);

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


test('ekipa24 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/ekipa24/1`);

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

test('rtvslo 1 novic', async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Navigate to the scraping route
  await page.goto(`http://localhost:${process.env.PORT}/news/scrape/rtvslo/1`);

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
  closeServer(); // Close the HTTP server
  await mongoose.disconnect();
}, 200000);

afterAll(async () => {
  try {
    closeServer(); // Close the HTTP server
    await mongoose.disconnect(); // Close the database connection
  } catch (error) {
    console.error(error);
  }
});
