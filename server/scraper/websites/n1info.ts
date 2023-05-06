import puppeteer from 'puppeteer';
import { INews } from '../../model/News';

async function n1infoSlovenija(n: number = 5) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  await page.goto('https://n1info.si/novice/slovenija');

  const links = await page.$$('a[href^="https://n1info.si/novice/slovenija"]');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const titleText = await link.evaluate((e) =>
      (e as HTMLElement).innerText.trim()
    );
    console.log('Title:', titleText);

    const url = await link.evaluate((e) => e.getAttribute('href'));
    console.log('URL:', url);

    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(url);

      const authorElement = await articlePage.$(
        'a[rel="author"].fn[target="_blank"]'
      );
      const authorText = await (authorElement
        ? authorElement.evaluate((e) => (e as HTMLElement).innerText.trim())
        : '');
      console.log('Author:', authorText);

      const contentElements = await articlePage.$$('p');
      const content = await Promise.all(
        contentElements.map((e) =>
          e.evaluate((el) => (el as HTMLElement).innerText.trim())
        )
      );
      const joinedContent = content.join(' ');
      console.log('Content:', joinedContent);

      const categoryLinks = await articlePage.$$(
        'a[href^="https://n1info.si/tag"]'
      );
      const categories = await Promise.all(
        categoryLinks.map((link) =>
          link.evaluate((e) => (e as HTMLElement).innerText.trim())
        )
      );
      console.log('Categories:', categories);

      news.push({
        title: titleText,
        url,
        authors: [authorText],
        date: new Date(),
        content: joinedContent,
        categories,
        location: 'Slovenija',
      });

      await articlePage.close();
    }
  }

  return news;
}

export = n1infoSlovenija;
