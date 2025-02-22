import puppeteer from 'puppeteer';
import { INews } from '../../model/News';
import * as Db from '../../db/db';

async function n1infoSlovenija(n: number = 5) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://n1info.si/novice/slovenija');

  const links = await page.$$('a[href^="https://n1info.si/novice/slovenija"]');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const titleText = await link.evaluate((e) =>
      (e as HTMLElement).innerText.trim()
    );

    const url = await link.evaluate((e) => e.getAttribute('href'));

    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(url);

      const authorElement = await articlePage.$(
        'a[rel="author"].fn[target="_blank"]'
      );
      const authorText = await (authorElement
        ? authorElement.evaluate((e) => (e as HTMLElement).innerText.trim())
        : '');

      const contentElements = await articlePage.$$('p');
      const content = await Promise.all(
        contentElements.map((e) =>
          e.evaluate((el) => (el as HTMLElement).innerText.trim())
        )
      );
      const joinedContent = content.join(' ');

      const categoryLinks = await articlePage.$$(
        'a[href^="https://n1info.si/tag"]'
      );
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
        url,
        authors: [authorText],
        date: new Date(),
        content: joinedContent,
        categories,
        views: [],
        location: {
          type: 'Point',
          coordinates: coords,
        },
      });

      await articlePage.close();
    }
  }

  return news;
}

export = n1infoSlovenija;
