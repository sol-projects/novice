import { Request, Response, NextFunction } from 'express';
const jwt = require('jsonwebtoken');
import { IUser, User } from '../model/User';

export enum Role {
  ReadOnly,
  Admin,
}

export async function authorization(
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (
    !req.headers.authorization ||
    req.headers.authorization.split(' ').length < 2
  ) {
    res
      .status(403)
      .json({
        message:
          'Access forbidden. You must provide authorization in the request header under Authorization: Bearer',
      });
  } else {
    const token = req.headers.authorization.split(' ')[1];
    const uuid = req.body.uuid;

    const user = await User.findOne({ uuid: process.env.LOGIN_UUID });
    if (!user) {
      return res.status(401).json({ message: 'Invalid UUID' });
    }

    try {
      const decoded = jwt.verify(token, user.public_key, {
        algorithms: ['RS256'],
      });

      if (decoded.role == 'admin') {
        next();
      } else {
        return res.status(401).json({ message: 'Invalid UUID' });
      }
    } catch (error) {
      return res.status(401).json({ message: 'Invalid token' });
    }
  }
}

export async function login(req: Request, res: Response) {
  try {
    const user = await User.findOne({ uuid: process.env.LOGIN_UUID });

    if (!user) {
      return res.status(401).json({ message: 'Invalid UUID' });
    }

    const token = jwt.sign(
      { sub: user.uuid, role: user.role },
      process.env.LOGIN_PRIVATE_KEY,
      { algorithm: 'RS256', expiresIn: '1h' }
    );

    return res.status(200).json({ token });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal server error' });
  }
}
