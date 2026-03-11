import { useRef, useEffect, useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";

const THRESHOLD = 70; // px of pull before triggering refresh
const MAX_PULL = 120; // max visual displacement

/**
 * Pull-to-refresh hook for mobile. Attach the returned ref to a scrollable
 * container. When the user pulls down from scrollTop=0, all TanStack queries
 * are invalidated. Returns pull state for rendering a spinner.
 */
export function usePullRefresh<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null);
  const queryClient = useQueryClient();
  const [pullDistance, setPullDistance] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const startY = useRef(0);
  const pulling = useRef(false);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await queryClient.invalidateQueries();
    // Small delay so spinner is visible
    await new Promise((r) => setTimeout(r, 400));
    setRefreshing(false);
    setPullDistance(0);
  }, [queryClient]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Find the scrollable ancestor (the content area)
    const scrollParent = el;

    function onTouchStart(e: TouchEvent) {
      if (refreshing) return;
      if (scrollParent.scrollTop > 5) return; // only from top
      startY.current = e.touches[0].clientY;
      pulling.current = true;
    }

    function onTouchMove(e: TouchEvent) {
      if (!pulling.current || refreshing) return;
      const dy = e.touches[0].clientY - startY.current;
      if (dy < 0) {
        pulling.current = false;
        setPullDistance(0);
        return;
      }
      // Dampened pull — feels natural
      const dampened = Math.min(MAX_PULL, dy * 0.4);
      setPullDistance(dampened);
    }

    function onTouchEnd() {
      if (!pulling.current) return;
      pulling.current = false;
      if (pullDistance >= THRESHOLD * 0.4) {
        onRefresh();
      } else {
        setPullDistance(0);
      }
    }

    scrollParent.addEventListener("touchstart", onTouchStart, { passive: true });
    scrollParent.addEventListener("touchmove", onTouchMove, { passive: true });
    scrollParent.addEventListener("touchend", onTouchEnd);

    return () => {
      scrollParent.removeEventListener("touchstart", onTouchStart);
      scrollParent.removeEventListener("touchmove", onTouchMove);
      scrollParent.removeEventListener("touchend", onTouchEnd);
    };
  }, [refreshing, pullDistance, onRefresh]);

  return { ref, pullDistance, refreshing };
}
