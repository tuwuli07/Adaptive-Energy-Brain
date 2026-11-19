// Friendly display labels; controller identifiers remain unchanged.
export function strategyName(name = '') {
  if (/oracle|perfect.foresight/i.test(name)) return 'Perfect-knowledge reference';
  if (/gnn/i.test(name)) return 'Forecast-led planning';
  if (/seasonal/i.test(name)) return 'Seasonal planning';
  if (/zero.action/i.test(name)) return 'Idle battery';
  if (/cvar/i.test(name)) return 'High-cost protection';
  if (/dro/i.test(name)) return 'Uncertainty protection';
  if (/ppo/i.test(name)) return 'Learned strategy';
  return name.replace(/ Baseline| \(16k steps\)/g, '');
}
