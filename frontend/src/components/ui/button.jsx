import React from 'react'

export function Button({ children, className = '', variant, ...rest }) {
  const cls = `${className}`
  return (
    <button className={cls} {...rest}>
      {children}
    </button>
  )
}

export default Button
