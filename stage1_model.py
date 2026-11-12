"""
ЕТАП 1. Імітаційна модель процесу обслуговування одним каналом
(клініка з одним стоматологом). Покрокова зміна модельного часу, крок 1 год.
"""
import numpy as np
import pandas as pd
 
from params import (N_STEPS, HOURS_PER_DAY, WORK_HOURS, QUEUE_LIMIT, SALARY, DAYS,
                    SERVICES, lam_at, outage_plan, new_client)
 
 
def simulate_one(rng, queue_limit=QUEUE_LIMIT, const_lambda=None, outages=True, log=False):
    """Один прогін моделі з одним стоматологом.
 
    На кожному кроці (1 робоча година):
      1) кількість нових клієнтів ~ Poisson(lambda періоду);
      2) кожен клієнт отримує послугу і тривалість ~ Exp(середнє);
      3) якщо зал очікування заповнений і лікар зайнятий – клієнт втрачений;
      4) якщо немає відключення, лікар має 1 год робочого часу:
         доробляє поточного пацієнта і бере з черги наступних.
    """
    plan = outage_plan(rng) if outages else np.zeros(N_STEPS, dtype=bool)
    queue = []            # черга: [(момент приходу, послуга, тривалість), ...]
    current = None        # пацієнт у кріслі: [залишок часу, послуга] або None
    served = lost = revenue = lost_revenue = 0
    idle_time = 0.0
    waits, rows = [], []
 
    for step in range(N_STEPS):
        hour = WORK_HOURS[step % HOURS_PER_DAY]
        lam = const_lambda if const_lambda is not None else lam_at(hour)
        off = bool(plan[step])
 
        # ---- надходження клієнтів
        k = rng.poisson(lam)
        lost_step = 0
        for _ in range(k):
            t_arr, s, dur = new_client(rng, step)
            if len(queue) >= queue_limit and (current is not None or off):
                lost += 1
                lost_step += 1
                lost_revenue += SERVICES["Ціна"].iat[s]
            else:
                queue.append((t_arr, s, dur))
        queue.sort(key=lambda x: x[0])
 
        # ---- обслуговування протягом години
        served_step = 0
        if not off:
            t, end = float(step), step + 1.0
            while t < end:
                if current is None:
                    if queue and queue[0][0] <= end:
                        t_arr, s, dur = queue.pop(0)
                        if t_arr > t:              # лікар чекає, поки прийде клієнт
                            idle_time += t_arr - t
                            t = t_arr
                        waits.append(t - t_arr)
                        current = [dur, s]
                    else:                          # нікого немає – простій
                        idle_time += end - t
                        break
                work = min(current[0], end - t)
                t += work
                current[0] -= work
                if current[0] <= 1e-9:             # пацієнта обслуговано
                    served += 1
                    served_step += 1
                    revenue += SERVICES["Ціна"].iat[current[1]]
                    current = None
 
        if log:
            rows.append({"Крок": step + 1, "День": step // HOURS_PER_DAY + 1,
                         "Година": f"{hour:02d}:00", "lambda": lam,
                         "Відключення": "так" if off else "", "Надійшло": k,
                         "Втрачено": lost_step, "Обслуговано": served_step,
                         "Черга": len(queue), "Лікар зайнятий": int(current is not None)})
 
    work_time = N_STEPS - plan.sum()
    total = served + lost + len(queue) + (current is not None)
    cost = SALARY * DAYS
    res = {"Стоматологів": 1, "Клієнтів": total, "Обслуговано": served, "Втрачено": lost,
           "P_відмови": lost / total if total else 0,
           "Сер. очікування, год": np.mean(waits) if waits else 0,
           "P_простою": idle_time / work_time if work_time else 0,
           "Завантаженість": 1 - idle_time / work_time if work_time else 0,
           "Годин відключення": int(plan.sum()),
           "Дохід": revenue, "Зарплата": cost, "Недоотримано": lost_revenue,
           "Прибуток": revenue - cost}
    return (res, pd.DataFrame(rows)) if log else res
 
 
if __name__ == "__main__":
    res, log = simulate_one(np.random.default_rng(42), log=True)
    print(log.head(16).to_string(index=False))
    print(res)
 