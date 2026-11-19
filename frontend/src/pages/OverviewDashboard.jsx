import React from 'react';
import { Link } from 'react-router-dom';
export default function OverviewDashboard() {
  return <section className="surface p-8 space-y-5"><span className="eyebrow">YOUR ENERGY WORKSPACE</span><h1>Energy intelligence for your site.</h1><p>Connect energy data for your home, building, business, renewable community, or microgrid.</p><Link className="button-primary" to="/data">Add energy data</Link></section>;
}
