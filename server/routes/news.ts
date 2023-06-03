import express, { Express, Request, Response } from 'express';
import * as Controller from '../controller/news';
import * as JWT from '../security/jwt';

const router = express.Router();

router.get('/', Controller.all);
router.get('/:id', Controller.id);
router.get('/scrape/:n', Controller.scrapeAll);
router.get('/scrape/:website/:n', Controller.scrape);

router.get('/categories/:categories', Controller.Filter.categories);
router.get('/authors/:authors', Controller.Filter.authors);
router.get('/location/:location', Controller.Filter.location);
router.get('/website/:website', Controller.Filter.website);
router.get('/date/before/:date', Controller.Filter.Date.before);
router.get('/date/after/:date', Controller.Filter.Date.after);
router.get('/date/after/:after/before/:before', Controller.Filter.Date.range);
router.get('/title/:title', Controller.Filter.title);
router.get('/content/:content', Controller.Filter.content);
//nizka prioriteta: dodaj za fromCoords in toCoords




router.put('/:id', JWT.authorization, Controller.update);
router.delete('/:id', JWT.authorization, Controller.remove);
router.post('/', JWT.authorization, Controller.add);
router.post('/views', Controller.view)
router.post('/store', JWT.authorization, Controller.store);
router.post('/login', JWT.login);
router.post('/geolang', Controller.geolang);

export = router;
