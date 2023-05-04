import { INews } from '../../model/News';
import puppeteer from 'puppeteer';

async function _siol(n: number) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  const date = new Date();
  await page.goto(
    `https://siol.net/pregled-dneva/${date.getFullYear()}-${
      date.getMonth() + 1
    }-${date.getDate()}`
  );

  const links = await page.$$('.card--timemachine > .card__link');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const url = `http://www.siol.net${await link.evaluate((e) =>
      e.getAttribute('href')
    )}`;

    const articlePage = await browser.newPage();
    await articlePage.goto(url);
    await articlePage.waitForSelector('.article__title', { visible: true });

    const title = await articlePage.$eval('.article__title', (element) =>
      element?.textContent?.trim()
    );

    if (!title) continue;

    const authors = await articlePage.$eval('.article__author', (e) =>
      (e as HTMLElement).innerText.trim().split(': ')[1].split(', ')
    );

    const dateUnparsed = await articlePage.$eval(
      '.article__publish_date--date',
      (e) => (e as HTMLElement).innerText.split(';')[0].trim()
    );

    const time = await articlePage.$eval('.article__publish_date--time', (e) =>
      (e as HTMLElement).innerText.trim().split('.')
    );

    const dateSplit = dateUnparsed.split('.');
    const date = new Date(
      +dateSplit[2],
      +dateSplit[1] - 1,
      +dateSplit[0] + 1,
      +time[0],
      +time[1],
      0
    );

    const content = await articlePage.$eval(
      '.article__main:not([entity="relatedArticle"])',
      (e) => (e as HTMLElement).innerText.trim()
    );
    const categories = await articlePage.$$eval('.article__tags--tag', (els) =>
      els.map((e) => (e as HTMLElement).innerText.trim())
    );

    news.push({
      title,
      url,
      date,
      authors,
      content,
      categories,
      location: '',
    });

    await articlePage.close();
  }

  await browser.close();

  return news;
}

export = _siol;
