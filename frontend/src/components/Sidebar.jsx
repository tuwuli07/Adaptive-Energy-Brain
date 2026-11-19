import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  TrendingUp,
  BatteryCharging,
  GitCompare,
  Compass,
  FileText,
  ShieldAlert,
  Sliders,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  Zap,
} from 'lucide-react';

export default function Sidebar({ collapsed, setCollapsed, darkMode, setDarkMode }) {
  const navItems = [
    { name: 'Overview', path: '/', icon: LayoutDashboard },
    { name: 'Energy Monitoring', path: '/energy', icon: Activity },
    { name: 'AI Forecasting', path: '/forecasting', icon: TrendingUp },
    { name: 'Battery Management', path: '/battery', icon: BatteryCharging },
    { name: 'Controller Benchmark', path: '/comparison', icon: GitCompare },
    { name: 'Risk Analytics', path: '/risk', icon: ShieldAlert },
    { name: 'Adaptive Control', path: '/adaptive', icon: Sliders },
    { name: 'Simulation Explorer', path: '/explorer', icon: Compass },
    { name: 'Research Results', path: '/research', icon: FileText },
    { name: 'About the Project', path: '/about', icon: BookOpen },
  ];

  return (
    <aside
      className={`fixed top-0 left-0 z-40 h-screen transition-all duration-300 ease-in-out border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col justify-between ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div>
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-100 dark:border-slate-800">
          <NavLink to="/" className="flex items-center gap-3 overflow-hidden">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 text-white shadow-md shadow-emerald-500/20 shrink-0">
              <Zap className="w-5 h-5" />
            </div>
            {!collapsed && (
              <div className="flex flex-col">
                <span className="font-bold text-sm tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap">
                  Adaptive Energy Brain
                </span>
                <span className="text-[10px] font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
                  Energy Management Platform
                </span>
              </div>
            )}
          </NavLink>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl font-medium text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-semibold shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800/60'
                } ${collapsed ? 'justify-center px-0' : ''}`
              }
              title={collapsed ? item.name : undefined}
            >
              <item.icon className="w-5 h-5 shrink-0" />
              {!collapsed && <span className="truncate">{item.name}</span>}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Footer Controls & Mode Toggle */}
      <div className="p-3 border-t border-slate-100 dark:border-slate-800 space-y-2">
        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${
            collapsed ? 'justify-center px-0' : ''
          }`}
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? (
            <>
              <Sun className="w-4 h-4 text-amber-400 shrink-0" />
              {!collapsed && <span>Light Theme</span>}
            </>
          ) : (
            <>
              <Moon className="w-4 h-4 text-indigo-500 shrink-0" />
              {!collapsed && <span>Dark Theme</span>}
            </>
          )}
        </button>

        {/* Collapse Sidebar Button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${
            collapsed ? 'justify-center px-0' : ''
          }`}
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>Collapse Sidebar</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
