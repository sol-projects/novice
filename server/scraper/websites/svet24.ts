import puppeteer from 'puppeteer';
import { INews } from '../../model/News';
async function noviceSvet24(n: number) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://novice.svet24.si/danes-objavljeno');

  const links = await page.$$('.divide-y a');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const titleElement = await link.$(
      'h3'
    );
    const titleText = await (titleElement
      ? titleElement.evaluate((e) => (e as HTMLElement).innerText.trim())
      : '');

    const authorElement = await link.$('.line-clamp-1');
    const authorText = await (authorElement
      ? authorElement.evaluate((e) => (e as HTMLElement).innerText.trim())
      : '');

      const url = await link.evaluate((e) => e.getAttribute('href'));


    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(`https://novice.svet24.si${url}`);

      const contentElements = await articlePage.$$('p');
      const content = await Promise.all(
        contentElements.map((e) =>
          e.evaluate((el) => (el as HTMLElement).innerText.trim())
        )
      );
      const joinedContent = content.join(' ');

      const dateElement = await link.$('.pr-4 flex items-center');
      const dateText = await (dateElement
        ? dateElement.evaluate((e) => (e as HTMLElement).innerText.trim())
        : '');

      news.push({
        title: titleText,
        url: `https://novice.svet24.si${url}`,
        date: new Date(),
        authors: [authorText],
        content: joinedContent,
        categories: [],
        views: [],
        location: {
          type: 'Point',
          coordinates: [0, 0],
        },
      });

      await articlePage.close();
    }
  }

  return news;
}

export = noviceSvet24;
