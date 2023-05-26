import puppeteer from 'puppeteer';
import { INews } from '../../model/News';
import * as Db from '../../db/db';

async function ekipaSvet24(n: number) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  await page.goto('https://ekipa.svet24.si/');

  const links = await page.$$('a[href^="/clanek"]');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const url = await link.evaluate((e) => e.getAttribute('href'));

    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(`https://ekipa.svet24.si${url}`);
      const titleElement = link;
      let titleText = '';
      if (titleElement) {
        titleText = await titleElement.evaluate((e) =>
          (e as HTMLElement).innerText.trim()
        );
      }
      const authors = await articlePage.$$eval('.top-author', (els) =>
        els.map((e) => {
          const text = (e as HTMLElement).innerText.trim();
          const [author] = text.split('\n');
          return author;
        })
      );

      const dateElement = await articlePage.$('.top-author');
      const dateText = await (dateElement
        ? dateElement.evaluate((e) => {
            const text = (e as HTMLElement).innerText.trim();
            const [, date] = text.split('\n');
            return date;
          })
        : '');

      const content = await articlePage.$eval('p', (e) =>
        (e as HTMLElement).innerText.trim()
      );

      const categoryLinks = await articlePage.$$('a[href^="/iskanje"]');
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
        title: titleText,
        url: `https://ekipa.svet24.si${url}`,
        date: new Date(dateText),
        authors,
        content,
        categories,
        location: {
            type: 'Point',
            coordinates: coords
        },
      });

      await articlePage.close();
    }
  }

  return news;
}

export = ekipaSvet24;
