import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import { render, screen, fireEvent } from '@testing-library/react';
import DynamicInput from './DynamicInput';

test('dynamic inputs notify once per edit, and parent rerenders do not notify again', () => {
  const change = jest.fn();
  const { rerender } = render(<ChakraProvider><DynamicInput onChange={change} /></ChakraProvider>);
  expect(change).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '+' }));
  expect(change).toHaveBeenLastCalledWith(['']);
  fireEvent.change(screen.getByPlaceholderText('Enter a value'), { target: { value: 'Ljubljana' } });
  expect(change).toHaveBeenLastCalledWith(['Ljubljana']);
  expect(change).toHaveBeenCalledTimes(2);
  rerender(<ChakraProvider><DynamicInput onChange={values => change(values)} /></ChakraProvider>);
  expect(change).toHaveBeenCalledTimes(2);
  fireEvent.click(screen.getByRole('button', { name: '-' }));
  expect(change).toHaveBeenLastCalledWith([]);
});
