declare module "@nivo/waffle" {
    import { ComponentType } from "react";
    import { DefaultSvgProps } from "@nivo/core";
  
    export interface WaffleDatum {
      id: string | number;
      value: number;
    }
  
    export interface WaffleSvgProps extends DefaultSvgProps {
      data: WaffleDatum[];
      total: number;
      rows: number;
      columns: number;
      fillDirection?: "top" | "right" | "bottom" | "left";
      padding?: number;
      cellSize?: number;
      margin?: DefaultSvgProps["margin"];
      emptyColor?: string;
      emptyOpacity?: number;
      colors?: DefaultSvgProps["colors"];
      borderColor?: DefaultSvgProps["borderColor"];
      borderWidth?: number;
      isInteractive?: boolean;
      tooltip?: ComponentType<WaffleCellProps>;
      legends?: LegendProps[];
      role?: string;
    }
  
    export const ResponsiveWaffle: ComponentType<WaffleSvgProps>;
  }
  