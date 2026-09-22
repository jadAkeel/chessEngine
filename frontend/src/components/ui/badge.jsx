import React from 'react'

export function Badge({ children, className = '', variant }) {
  const cls = `${className}`
  return <span className={cls}>{children}</span>
}

export default Badge
