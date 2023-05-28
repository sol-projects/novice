import express, { Request, Response } from 'express';
import websites from '../scraper/websites';
import { INews, News } from '../model/News';
import * as Socket from '../socket/socket';

async function run_query(res: Response, query: any) {
  try {
    const news = await News.find(query).exec();
    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to retrieve news from MongoDB');
  }
}

export async function id(req: Request, res: Response) {
  const { id } = req.params;

  try {
    const news = await News.findById(id);

    if (!news) {
      return res.status(404).json({ message: `News with id ${id} not found` });
    }

    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Server error');
  }
}

export async function remove(req: Request, res: Response) {
  const id = req.params.id;

  try {
    const deletedNews = await News.findByIdAndDelete(id);

    if (!deletedNews) {
      return res.status(404).send('News not found');
    }

    res.send(`News with ID ${id} deleted`);
  } catch (error) {
    console.error(error);
    res.status(500).send(`Failed to delete news with ID ${id}`);
  }
}
export async function update(req: Request, res: Response) {
  const id = req.params.id; // get the id from the route parameter
  const updatedNewsData = req.body; // get the updated news data directly from the request body

  if (!id) {
    return res.status(400).send('ID is required for updating news.');
  }

  try {
    // Update the news in the database
    const updatedNews = await News.findByIdAndUpdate(
      id,
      updatedNewsData,
      { new: true, useFindAndModify: false } // This option returns the updated document
    );

    if (!updatedNews) {
      return res.status(404).json({ message: `News with ID ${id} not found` });
    }

    res.json(updatedNews);

  } catch (error) {
    console.error(error);
    res.status(500).send('Server error while updating news');
  }
}


export async function all(req: Request, res: Response) {
  run_query(res, {});
}
/*
export async function store(req: Request, res: Response) {
  let news: INews[] = [];
  for await (let [key, value] of websites) {
    const result = await value(req.body.n);
    console.log(`Evaluating website ${key} before pushing to database...`);
    for (let value of result) {
      const existingNews = await News.findOne({
        title: value.title,
        content: value.content,
      });

      if (!existingNews) {
        news.push(value);
      } else {
        console.log(
          `Article "${value.title}" on website ${key} already exists. Not pushing to database...`
        );
      }
    }
    console.log(`Website ${key} evaluated successfully...`);
  }

  news.sort((a, b) => a.date.getTime() - b.date.getTime());
  try {
    await News.create(news);
    Socket.emit('news-added', news);
    res.status(201).json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to save news to MongoDB');
  }
} */
export async function store(req: Request, res: Response) {
  let news: INews[] = [];
  
  let payload = req.body; // assuming req.body is a single news item

  if (Array.isArray(payload)) {
    payload = payload[0]; // If an array is received, we take the first object
  }

  console.log(`Processing input before pushing to database...`);

  const value = payload;
  const existingNews = await News.findOne({
    title: value.title,
    content: value.content,
  });

  if (!existingNews) {
    news.push(value);
  } else {
    console.log(
      `Article "${value.title}" already exists. Not pushing to database...`
    );
  }
  
  console.log(`Input processed successfully...`);
  
  try {
    await News.create(news);
    Socket.emit('news-added', news);
    res.status(201).json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to save news to MongoDB');
  }
}



export async function scrape(req: Request, res: Response) {
  let news: INews[] = [];
  const website = await websites.get(req.params.website);

  if (!website) {
    res
      .status(500)
      .send(`Server error. Website with the name ${website} does not exist.`);
  } else {
    const result = await website(+req.params.n);

    for (let value of result) {
      news.push(value);
    }

    res.json(news);
  }
}

export async function scrapeAll(req: Request, res: Response) {
  let news: INews[] = [];

  for await (let [key, value] of websites) {
    const valueResult = await value(+req.params.n);
    for (let value of valueResult) {
      news.push(value);
    }
  }

  res.json(news);
}

export namespace Filter {
  export async function categories(req: Request, res: Response) {
    run_query(res, { categories: { $all: req.params.categories.split(',') } });
  }

  export async function authors(req: Request, res: Response) {
    run_query(res, { authors: { $all: req.params.authors.split(',') } });
  }

  export function location(req: Request, res: Response) {
    run_query(res, { location: req.params.location });
  }

  export function website(req: Request, res: Response) {
    run_query(res, { url: { $regex: req.params.website, $options: 'i' } });
  }

  export function title(req: Request, res: Response) {
    run_query(res, { title: { $regex: req.params.title, $options: 'i' } });
  }

  export function content(req: Request, res: Response) {
    run_query(res, { content: { $regex: req.params.content, $options: 'i' } });
  }

  export namespace Date {
    export function before(req: Request, res: Response) {
      run_query(res, { date: { $lt: req.params.date } });
    }

    export function after(req: Request, res: Response) {
      run_query(res, { date: { $gt: req.params.date } });
    }

    export function range(req: Request, res: Response) {
      run_query(res, {
        date: {
          $gte: req.params.after,
          $lte: req.params.before,
        },
      });
    }
  }
}
