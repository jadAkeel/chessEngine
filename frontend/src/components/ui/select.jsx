import React, { createContext, useContext, useMemo } from 'react'

const SelectContext = createContext({ value: undefined, onValueChange: () => {}, items: [] })

export function Select({ value, onValueChange, children }) {
  const items = []
  React.Children.forEach(children, (child) => {
    if (!child) return
    if (child.type && child.type.displayName === 'SelectContent') {
      React.Children.forEach(child.props.children, (item) => {
        if (item && item.type && item.type.displayName === 'SelectItem') {
          items.push(item.props)
        }
      })
    }
  })

  const contextValue = useMemo(() => ({ value, onValueChange, items }), [value, onValueChange, items])
  return <SelectContext.Provider value={contextValue}>{children}</SelectContext.Provider>
}

export function SelectTrigger({ className = '' }) {
  const ctx = useContext(SelectContext)
  return (
    <select
      className={className}
      value={ctx.value}
      onChange={(e) => ctx.onValueChange && ctx.onValueChange(e.target.value)}
    >
      {ctx.items.map((it) => (
        <option key={it.value} value={it.value}>
          {it.children}
        </option>
      ))}
    </select>
  )
}

export function SelectValue() {
  return null
}

export function SelectContent() {
  return null
}

export function SelectItem() {
  return null
}

SelectContent.displayName = 'SelectContent'
SelectItem.displayName = 'SelectItem'

export default Select
