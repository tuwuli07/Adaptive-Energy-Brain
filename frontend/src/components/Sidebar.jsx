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
  Bot,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  Zap,
  UploadCloud,
} from 'lucide-react';

export default function Sidebar({ collapsed, setCollapsed, darkMode, setDarkMode }) {
  const navigationGroups = [
    { title: 'Workspace', items: [
      { name: 'Dashboard', path: '/', icon: LayoutDashboard },
      { name: 'Energy use', path: '/energy', icon: Activity },
      { name: 'My Energy Data', path: '/data', icon: UploadCloud },
    ] },
    { title: 'Energy planning', items: [
      { name: 'Forecast', path: '/forecasting', icon: TrendingUp },
      { name: 'Battery storage', path: '/battery', icon: BatteryCharging },
      { name: 'Smart control', path: '/adaptive', icon: Sliders },
    ] },
    { title: 'Explore further', items: [
      { name: 'Study & methods', path: '/research', icon: BookOpen },
    ] },
  ];

  return (
    <aside
      className={`app-sidebar fixed top-0 left-0 z-40 h-screen transition-all duration-300 ease-in-out border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col justify-between ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div>
        <div className="h-20 flex items-center justify-between px-4 border-b border-slate-100 dark:border-slate-800">
          <NavLink to="/" className="flex items-center gap-3 overflow-hidden">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-blue-600 to-sky-400 text-white shadow-md shadow-blue-500/20 shrink-0">
              <Zap className="w-5 h-5" />
            </div>
            {!collapsed && (
              <div className="flex flex-col">
                <span className="font-extrabold text-sm tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap">
                  Adaptive Energy Brain
                </span>
                <span className="text-[10px] font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                  Energy intelligence
                </span>
              </div>
            )}
          </NavLink>
        </div>

        {/* Grouped Navigation */}
        <nav className="p-3 space-y-3 overflow-y-auto max-h-[calc(100dvh-220px)]">
          {navigationGroups.map((group) => (
            <div key={group.title} className="space-y-1">
              {!collapsed && (
                <div className="px-3 py-1 text-[10px] font-bold tracking-wider text-slate-400 dark:text-slate-500 uppercase">
                  {group.title}
                </div>
              )}
              {collapsed && (
                <div className="w-6 h-px bg-slate-200 dark:bg-slate-800 mx-auto my-2" />
              )}
              {group.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-xl font-medium text-xs sm:text-sm transition-all duration-150 ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/70 dark:text-blue-300 font-semibold shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800/60'
                    } ${collapsed ? 'justify-center px-0' : ''}`
                  }
                  aria-label={item.name}
                  title={item.name}
                >
                  <item.icon className="w-4 h-4 sm:w-4.5 sm:h-4.5 shrink-0" />
                  {!collapsed && <span className="truncate">{item.name}</span>}
                </NavLink>
              ))}
            </div>
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
          aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
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

        {/* Collapse menu Button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${
            collapsed ? 'justify-center px-0' : ''
          }`}
          aria-label={collapsed ? 'Expand menu' : 'Collapse menu'}
          title={collapsed ? 'Expand Sidebar' : 'Collapse menu'}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>Collapse menu</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
