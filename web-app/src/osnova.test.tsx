import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

import { App } from './App';

describe('App', () => {
  it('should render the app component', () => {
    render(<App />);
    expect(screen.getByText('Your app content')).toBeInTheDocument();
  });

  it('should handle button click', () => {
    render(<App />);
    const button = screen.getByRole('a');
    fireEvent.click(button);
    expect(screen.getByText('Button clicked!')).toBeInTheDocument();
  });
});
