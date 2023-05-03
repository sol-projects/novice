import puppeteer from 'puppeteer';
import { INews } from '../../model/News';

async function _24ur(n: number) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://www.24ur.com/novice');

  const links = await page.$$('a[href^="/novice/"]');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];
    const titleElement = await link.$(
      'div h1, div h2, div h3, div h4, div h5, div h6'
    );
    const titleText = await (titleElement
      ? titleElement.evaluate((e) => e.innerText)
      : '');
    const altTitleElement = await link.$(
      'h1 span, h2 span, h3 span, h4 span, h5 span, h6 span'
    );
    const altTitleText = await (altTitleElement
      ? altTitleElement.evaluate((e) => e.innerText)
      : '');

    const title = (
      titleText.trim().length > 0 ? titleText : altTitleText
    ).trim();
    if (!title) continue;

    const url = `http://www.24ur.com${await link.evaluate((e) =>
      e.getAttribute('href')
    )}`;

    const articlePage = await browser.newPage();
    await articlePage.goto(url);
    await articlePage.waitForSelector('.article__body', { visible: true });

    const authors = await articlePage.$$eval('.article__author', (els) =>
      els.map((e) => (e as HTMLElement).innerText.trim())
    );
    const locationDateUnparsed = await articlePage.$eval(
      '.leading-caption',
      (e) => (e as HTMLElement).innerText.trim()
    );
    const locationDate = locationDateUnparsed.split(', ');
    const dateSplit = locationDate[1].split('. ');
    const date = new Date(
      +dateSplit[2],
      +dateSplit[1] - 1,
      +dateSplit[0] + 1,
      0,
      0,
      0
    );
    const content = await articlePage.$eval('.article__body', (e) =>
      (e as HTMLElement).innerText.trim()
    );

    news.push({
      title,
      url,
      date,
      authors,
      content,
      categories: [],
      location: locationDate[0],
    });

    await articlePage.close();
  }

  await browser.close();

  return news;
}

export = _24ur;
