import puppeteer from 'puppeteer';
import { INews } from '../../model/News';
import * as Db from '../../db/db';

async function dnevnik(n: number) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://www.dnevnik.si/slovenija');

  const links = await page.$$('a[href^="/104"].view-more');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const url = await link.evaluate((e) => e.getAttribute('href'));

    if (url?.startsWith('/104')) {
      const articlePage = await browser.newPage();
      await articlePage.goto(`https://www.dnevnik.si${url}`);
      //await articlePage.waitForSelector('.entry-content', { visible: true });

      const authors = await articlePage.$$eval('.article-source', (els) =>
        els.map((e) => (e as HTMLElement).innerText.split(',')[0].trim())
      );

      const dateElement = await articlePage.$('.dtstamp');
      const dateText = await (dateElement
        ? dateElement.evaluate((e) => (e as HTMLElement).innerText)
        : '');
      const date = new Date(dateText);

      const content = await articlePage.$eval('article', (e) =>
        (e as HTMLElement).innerText.trim()
      );

      const firstSentence = content.split('\n')[0];
      const title = firstSentence.endsWith('.')
        ? firstSentence
        : `${firstSentence}.`;
      const categoryLinks = await articlePage.$$('a[href*="/tag/"]');
      const categories = await Promise.all(
        categoryLinks.map((link) =>
          link.evaluate((e) =>
            (e as HTMLElement).innerText.trim().toLowerCase()
          )
        )
      );

    const location = Db.Util.getFirstSettlement(categories);
    const coords = await Db.Util.toCoords(location);

    news.push({
        title: title.trim(),
        url: `https://www.dnevnik.si${url}`,
        date: new Date(),
        authors,
        content,
        categories,
        views: [],
        location: {
            type: 'Point',
            coordinates: coords
        },
      });
    }
  }

  await browser.close();

  return news;
}

export = dnevnik;
